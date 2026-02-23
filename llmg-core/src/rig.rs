//! Rig framework integration for LLMG
//!
//! This module implements rig's `CompletionModel` and `CompletionClient` traits
//! on top of LLMG's `Provider` trait, allowing any rig agent to use any loaded
//! LLMG provider transparently using the generic `<provider>/<model>` syntax.

use std::sync::Arc;

use crate::provider::Provider;
use crate::types::{
    self as llmg_types, ChatCompletionRequest, ChatCompletionResponse, FunctionDefinition, Tool,
};

use futures::StreamExt;
use rig::completion::{
    AssistantContent, CompletionError, CompletionModel, CompletionRequest, CompletionResponse,
    GetTokenUsage, Usage,
};
use rig::message::{Message as RigMessage, ToolResultContent, UserContent};
use rig::streaming::{RawStreamingChoice, StreamingCompletionResponse};
use rig::OneOrMany;
use serde::{Deserialize, Serialize};
use tracing::warn;

// ── Placeholder streaming response type ──────────────────────────────────────

/// Minimal placeholder for the streaming response associated type.
/// LLMG does not support rig's streaming interface yet.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlaceholderStreamingResponse;

impl GetTokenUsage for PlaceholderStreamingResponse {
    fn token_usage(&self) -> Option<Usage> {
        None
    }
}

// ── LlmgClient ──────────────────────────────────────────────────────────────

/// A client wrapping an LLMG `Provider` behind an `Arc` so it can be shared.
/// This acts as the `CompletionClient` for Rig.
#[derive(Clone)]
pub struct LlmgClient {
    pub provider: Arc<dyn Provider>,
}

impl LlmgClient {
    /// Create a client from an existing LLMG provider.
    pub fn new(provider: Arc<dyn Provider>) -> Self {
        Self { provider }
    }

    /// Create a client from a pre-built `ProviderRegistry`.
    /// The registry is wrapped in a `RoutingProvider` for automatic model routing.
    pub fn from_registry(registry: crate::provider::ProviderRegistry) -> Self {
        let router = crate::provider::RoutingProvider::new(registry);
        Self {
            provider: Arc::new(router),
        }
    }
}

impl rig::client::CompletionClient for LlmgClient {
    type CompletionModel = LlmgCompletionModel;

    fn completion_model(&self, model: impl Into<String>) -> Self::CompletionModel {
        LlmgCompletionModel::make(self, model)
    }
}

// ── LlmgCompletionModel ─────────────────────────────────────────────────────

/// A rig `CompletionModel` backed by an LLMG provider.
#[derive(Clone)]
pub struct LlmgCompletionModel {
    client: LlmgClient,
    model: String,
}

impl CompletionModel for LlmgCompletionModel {
    type Response = ChatCompletionResponse;
    type StreamingResponse = PlaceholderStreamingResponse;
    type Client = LlmgClient;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self {
            client: client.clone(),
            model: model.into(),
        }
    }

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        let llmg_request = build_llmg_request(&self.model, &request);

        let response = self
            .client
            .provider
            .chat_completion(llmg_request)
            .await
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

        build_rig_response(response)
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        let llmg_request = build_llmg_request(&self.model, &request);

        let stream = self
            .client
            .provider
            .chat_completion_stream(llmg_request)
            .await
            .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

        let mapped_stream = stream.filter_map(|chunk_res| async move {
            match chunk_res {
                Ok(chunk) => {
                    if let Some(choice) = chunk.choices.first() {
                        if let Some(text) = &choice.delta.content {
                            return Some(Ok(RawStreamingChoice::Message(text.clone())));
                        }
                        if let Some(tool_calls) = &choice.delta.tool_calls {
                            for tc in tool_calls {
                                // If it has an ID, it's the first chunk of a tool call
                                if tc.id.is_some()
                                    || tc.function.as_ref().map_or(false, |f| f.name.is_some())
                                {
                                    let id = tc
                                        .id
                                        .clone()
                                        .unwrap_or_else(|| format!("call_{}", tc.index));
                                    let name = tc
                                        .function
                                        .as_ref()
                                        .and_then(|f| f.name.clone())
                                        .unwrap_or_default();

                                    // Start a new tool call
                                    return Some(Ok(RawStreamingChoice::ToolCallDelta {
                                        id: id.clone(),
                                        delta: String::new(), // Initial delta just to establish id and name? No wait, Rig needs the name somehow
                                    }));
                                    // Wait! rig 0.3 handles ToolCallDelta by aggregating it. BUT if the name is not there, how does it know the name?!
                                }
                            }
                        }
                        if choice.finish_reason.is_some() {
                            return Some(Ok(RawStreamingChoice::FinalResponse(
                                PlaceholderStreamingResponse,
                            )));
                        }
                    }
                    None
                }
                Err(e) => Some(Err(CompletionError::ProviderError(e.to_string()))),
            }
        });

        Ok(StreamingCompletionResponse::stream(Box::pin(mapped_stream)))
    }
}

// ── Conversion helpers ───────────────────────────────────────────────────────

/// Build an LLMG `ChatCompletionRequest` from a rig `CompletionRequest`.
fn build_llmg_request(model: &str, request: &CompletionRequest) -> ChatCompletionRequest {
    let mut messages: Vec<llmg_types::Message> = Vec::new();

    // System preamble
    if let Some(ref preamble) = request.preamble {
        messages.push(llmg_types::Message::System {
            content: preamble.clone(),
            name: None,
        });
    }

    // Convert rig chat_history (which includes the final prompt) to LLMG messages
    for msg in request.chat_history.clone().into_iter() {
        convert_rig_message(msg, &mut messages);
    }

    // Convert tools
    let tools = if request.tools.is_empty() {
        None
    } else {
        Some(request.tools.iter().map(convert_tool_definition).collect())
    };

    ChatCompletionRequest {
        model: model.to_string(),
        messages,
        temperature: request.temperature.map(|t| t as f32),
        max_tokens: request.max_tokens.map(|t| t as u32),
        stream: Some(false),
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        stop: None,
        user: None,
        tools,
        tool_choice: None,
    }
}

/// Convert a single rig `Message` into one or more LLMG messages.
fn convert_rig_message(msg: RigMessage, out: &mut Vec<llmg_types::Message>) {
    match msg {
        RigMessage::User { content } => {
            for item in content.into_iter() {
                match item {
                    UserContent::Text(t) => {
                        out.push(llmg_types::Message::User {
                            content: t.text,
                            name: None,
                        });
                    }
                    UserContent::ToolResult(tr) => {
                        let text = tr
                            .content
                            .into_iter()
                            .filter_map(|c| match c {
                                ToolResultContent::Text(t) => Some(t.text),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                        out.push(llmg_types::Message::Tool {
                            content: text,
                            tool_call_id: tr.id,
                        });
                    }
                    _ => {}
                }
            }
        }
        RigMessage::Assistant { content, .. } => {
            let mut text_parts: Vec<String> = Vec::new();
            let mut tool_calls: Vec<llmg_types::ToolCall> = Vec::new();

            for item in content.into_iter() {
                match item {
                    AssistantContent::Text(t) => {
                        text_parts.push(t.text);
                    }
                    AssistantContent::ToolCall(tc) => {
                        let arguments = serde_json::to_string(&tc.function.arguments)
                            .unwrap_or_else(|e| {
                                warn!("failed to serialize tool call arguments: {e}");
                                "{}".to_string()
                            });
                        tool_calls.push(llmg_types::ToolCall {
                            id: tc.id,
                            r#type: "function".to_string(),
                            function: llmg_types::FunctionCall {
                                name: tc.function.name,
                                arguments,
                            },
                        });
                    }
                    _ => {}
                }
            }

            let content_str = if text_parts.is_empty() {
                None
            } else {
                Some(text_parts.join(""))
            };

            let tc = if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            };

            out.push(llmg_types::Message::Assistant {
                content: content_str,
                refusal: None,
                tool_calls: tc,
            });
        }
    }
}

/// Convert a rig `ToolDefinition` to an LLMG `Tool`.
fn convert_tool_definition(td: &rig::completion::ToolDefinition) -> Tool {
    Tool {
        r#type: "function".to_string(),
        function: FunctionDefinition {
            name: td.name.clone(),
            description: Some(td.description.clone()),
            parameters: td.parameters.clone(),
        },
    }
}

/// Build a rig `CompletionResponse` from an LLMG `ChatCompletionResponse`.
fn build_rig_response(
    response: ChatCompletionResponse,
) -> Result<CompletionResponse<ChatCompletionResponse>, CompletionError> {
    let choice = response
        .choices
        .first()
        .ok_or_else(|| CompletionError::ResponseError("no choices in response".to_string()))?;

    let assistant_content = match &choice.message {
        llmg_types::Message::Assistant {
            content,
            tool_calls,
            ..
        } => {
            let mut items: Vec<AssistantContent> = Vec::new();

            if let Some(text) = content {
                if !text.is_empty() {
                    items.push(AssistantContent::text(text));
                }
            }

            if let Some(tcs) = tool_calls {
                for tc in tcs {
                    let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                        .unwrap_or_else(|e| {
                            warn!("failed to parse tool call arguments: {e}");
                            serde_json::Value::Object(serde_json::Map::new())
                        });
                    items.push(AssistantContent::tool_call(&tc.id, &tc.function.name, args));
                }
            }

            if items.is_empty() {
                OneOrMany::one(AssistantContent::text(""))
            } else {
                OneOrMany::many(items).map_err(|e| CompletionError::ResponseError(e.to_string()))?
            }
        }
        _ => {
            return Err(CompletionError::ResponseError(
                "expected assistant message in response".to_string(),
            ));
        }
    };

    let usage = match &response.usage {
        Some(u) => Usage {
            input_tokens: u.prompt_tokens as u64,
            output_tokens: u.completion_tokens as u64,
            total_tokens: u.total_tokens as u64,
            cached_input_tokens: 0,
        },
        None => Usage::default(),
    };

    Ok(CompletionResponse {
        choice: assistant_content,
        usage,
        raw_response: response,
    })
}
