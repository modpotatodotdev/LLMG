#![allow(clippy::if_same_then_else, clippy::redundant_closure)]

use eventsource_stream::Eventsource;
use futures::{StreamExt, TryStreamExt};
use llmg_core::{
    provider::{ApiKeyCredentials, ChatCompletionStream, Credentials, LlmError, Provider},
    streaming::{ChatCompletionChunk, ChoiceDelta, DeltaContent},
    types::{
        ChatCompletionRequest, ChatCompletionResponse, Choice, EmbeddingRequest, EmbeddingResponse,
        Message, Usage,
    },
};
use std::future::Future;
use std::pin::Pin;

/// Anthropic Claude API client
#[derive(Debug)]
pub struct AnthropicClient {
    http_client: reqwest::Client,
    base_url: String,
    credentials: Box<dyn Credentials>,
}

/// Anthropic-specific request format
#[derive(Debug, serde::Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
}

#[derive(Debug, serde::Serialize)]
pub struct AnthropicTool {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

/// Anthropic message format
#[derive(Debug, serde::Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

/// Anthropic response format
#[derive(Debug, serde::Deserialize)]
struct AnthropicResponse {
    id: String,
    content: Vec<AnthropicContent>,
    model: String,
    usage: AnthropicUsage,
}

#[derive(Debug, serde::Deserialize)]
struct AnthropicContent {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
}

#[derive(Debug, serde::Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

impl AnthropicClient {
    /// Create a new Anthropic client from environment
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| LlmError::AuthError)?;

        Ok(Self::new(api_key))
    }

    /// Create a new Anthropic client with explicit API key
    pub fn new(api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();

        Self {
            http_client: reqwest::Client::new(),
            base_url: "https://api.anthropic.com/v1".to_string(),
            credentials: Box::new(ApiKeyCredentials::with_header(api_key, "x-api-key")),
        }
    }

    /// Create with custom base URL
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Convert OpenAI format to Anthropic format
    fn convert_request(&self, request: ChatCompletionRequest) -> AnthropicRequest {
        let mut system = None;
        let mut messages = Vec::new();

        for msg in request.messages {
            match msg {
                Message::System { content, .. } => {
                    system = Some(content);
                }
                Message::User { content, .. } => {
                    messages.push(AnthropicMessage {
                        role: "user".to_string(),
                        content,
                    });
                }
                Message::Assistant {
                    content: Some(content),
                    ..
                } => {
                    messages.push(AnthropicMessage {
                        role: "assistant".to_string(),
                        content,
                    });
                }
                _ => {} // Ignore other message types for now
            }
        }

        // Handle structured outputs via prompt engineering for providers without native support
        if let Some(ref response_format) = request.response_format {
            if response_format.is_structured() {
                let schema_instruction = if let Some(schema_config) = response_format.schema() {
                    format!(
                        "\n\nYou must respond with JSON that strictly adheres to this schema:\n{}",
                        serde_json::to_string_pretty(&schema_config.schema).unwrap_or_default()
                    )
                } else {
                    "\n\nYou must respond with valid JSON only.".to_string()
                };

                system = Some(format!(
                    "{}{}",
                    system.unwrap_or_default(),
                    schema_instruction
                ));
            }
        }

        AnthropicRequest {
            model: request.model,
            messages,
            system,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            stop_sequences: request.stop,
            stream: request.stream,
            tools: request.tools.map(|tools| {
                tools
                    .into_iter()
                    .map(|tool| AnthropicTool {
                        name: tool.function.name,
                        description: tool.function.description,
                        input_schema: if tool.function.parameters.is_null() {
                            serde_json::json!({"type": "object", "properties": {}})
                        } else {
                            tool.function.parameters
                        },
                    })
                    .collect()
            }),
            tool_choice: request.tool_choice.map(|choice| match choice {
                llmg_core::types::ToolChoice::String(s) => match s.as_str() {
                    "auto" => serde_json::json!({"type": "auto"}),
                    "none" => serde_json::json!({"type": "none"}),
                    "required" => serde_json::json!({"type": "any"}),
                    _ => serde_json::json!({"type": "auto"}),
                },
                llmg_core::types::ToolChoice::Named(named) => serde_json::json!({
                    "type": "tool",
                    "name": named.function.name
                }),
            }),
        }
    }

    /// Convert Anthropic response to OpenAI format
    fn convert_response(&self, response: AnthropicResponse) -> ChatCompletionResponse {
        let content = response
            .content
            .into_iter()
            .map(|c| c.text)
            .collect::<Vec<_>>()
            .join("");

        ChatCompletionResponse {
            id: response.id,
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: response.model,
            choices: vec![Choice {
                index: 0,
                message: Message::Assistant {
                    content: Some(content),
                    refusal: None,
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(Usage {
                prompt_tokens: response.usage.input_tokens,
                completion_tokens: response.usage.output_tokens,
                total_tokens: response.usage.input_tokens + response.usage.output_tokens,
            }),
        }
    }

    async fn make_request(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        let anthropic_req = self.convert_request(request);
        let url = format!("{}/messages", self.base_url);

        let mut req = self
            .http_client
            .post(&url)
            .json(&anthropic_req)
            .header("anthropic-version", "2023-06-01")
            .build()
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        self.credentials.apply(&mut req)?;

        let response = self
            .http_client
            .execute(req)
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError {
                status,
                message: text,
            });
        }

        let anthropic_resp: AnthropicResponse = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        Ok(self.convert_response(anthropic_resp))
    }

    async fn make_stream_request(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionStream, LlmError> {
        let anthropic_req = self.convert_request(request);
        let mut stream_request = anthropic_req;
        stream_request.stream = Some(true);

        let url = format!("{}/messages", self.base_url);

        let mut req = self
            .http_client
            .post(&url)
            .json(&stream_request)
            .header("anthropic-version", "2023-06-01")
            .build()
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        self.credentials.apply(&mut req)?;

        let response = self
            .http_client
            .execute(req)
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError {
                status,
                message: text,
            });
        }

        let chunk_id = ChatCompletionChunk::generate_id();
        let model = stream_request.model.clone();

        let stream = response
            .bytes_stream()
            .eventsource()
            .map_err(|e| LlmError::HttpError(e.to_string()))
            .then(move |event_result| {
                let chunk_id = chunk_id.clone();
                let model = model.clone();
                async move {
                    match event_result {
                        Ok(event) => parse_anthropic_sse_data(&event.data, &chunk_id, &model),
                        Err(e) => Err(LlmError::HttpError(e.to_string())),
                    }
                }
            })
            .try_filter_map(|chunk| async move { Ok(chunk) });

        Ok(Box::pin(stream) as ChatCompletionStream)
    }
}

#[async_trait::async_trait]
impl Provider for AnthropicClient {
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        self.make_request(request).await
    }

    fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ChatCompletionStream, LlmError>> + Send + '_>> {
        Box::pin(self.make_stream_request(request))
    }

    async fn embeddings(&self, _request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        Err(LlmError::ProviderError(
            "Anthropic does not support embeddings".to_string(),
        ))
    }
    fn provider_name(&self) -> &'static str {
        "anthropic"
    }
}

fn parse_anthropic_sse_data(
    data: &str,
    chunk_id: &str,
    model: &str,
) -> Result<Option<ChatCompletionChunk>, LlmError> {
    let data = data.trim();
    if data.is_empty() {
        return Ok(None);
    }

    let parsed: serde_json::Value =
        serde_json::from_str(data).map_err(LlmError::SerializationError)?;

    let event_type = parsed.get("type").and_then(|t| t.as_str()).unwrap_or("");

    match event_type {
        "content_block_delta" => {
            let delta = parsed.get("delta");
            let text = delta
                .and_then(|d| d.get("text"))
                .and_then(|t| t.as_str())
                .map(|s| s.to_string());

            let index = parsed.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as u32;

            if let Some(content) = text {
                return Ok(Some(ChatCompletionChunk {
                    id: chunk_id.to_string(),
                    object: "chat.completion.chunk".to_string(),
                    created: chrono::Utc::now().timestamp(),
                    model: model.to_string(),
                    choices: vec![ChoiceDelta {
                        index,
                        delta: DeltaContent::content(content),
                        finish_reason: None,
                    }],
                    usage: None,
                }));
            }
        }
        "content_block_start" => {
            let content_block = parsed.get("content_block");
            let index = parsed.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as u32;

            if content_block
                .and_then(|cb| cb.get("type"))
                .and_then(|t| t.as_str())
                == Some("text")
            {
                return Ok(Some(ChatCompletionChunk {
                    id: chunk_id.to_string(),
                    object: "chat.completion.chunk".to_string(),
                    created: chrono::Utc::now().timestamp(),
                    model: model.to_string(),
                    choices: vec![ChoiceDelta {
                        index,
                        delta: DeltaContent::role(),
                        finish_reason: None,
                    }],
                    usage: None,
                }));
            }
        }
        "message_delta" => {
            let delta = parsed.get("delta");
            let stop_reason = delta
                .and_then(|d| d.get("stop_reason"))
                .and_then(|s| s.as_str());

            if let Some(reason) = stop_reason {
                let finish_reason = match reason {
                    "end_turn" => "stop",
                    "max_tokens" => "length",
                    _ => reason,
                };

                return Ok(Some(ChatCompletionChunk {
                    id: chunk_id.to_string(),
                    object: "chat.completion.chunk".to_string(),
                    created: chrono::Utc::now().timestamp(),
                    model: model.to_string(),
                    choices: vec![ChoiceDelta {
                        index: 0,
                        delta: DeltaContent::empty(),
                        finish_reason: Some(finish_reason.to_string()),
                    }],
                    usage: None,
                }));
            }
        }
        "message_stop" => {
            return Ok(Some(ChatCompletionChunk::final_chunk(
                chunk_id.to_string(),
                model.to_string(),
                "stop",
            )));
        }
        _ => {}
    }

    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anthropic_client_creation() {
        let client = AnthropicClient::new("test-key");
        assert_eq!(client.provider_name(), "anthropic");
    }

    #[test]
    fn test_request_conversion() {
        let client = AnthropicClient::new("test-key");

        let request = ChatCompletionRequest {
            model: "claude-3-opus".to_string(),
            messages: vec![
                Message::System {
                    content: "You are a helpful assistant".to_string(),
                    name: None,
                },
                Message::User {
                    content: "Hello!".to_string(),
                    name: None,
                },
            ],
            temperature: None,
            max_tokens: Some(100),
            stream: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            user: None,
            tools: None,
            tool_choice: None,
            response_format: None,
        };

        let anthropic_req = client.convert_request(request);

        assert_eq!(anthropic_req.model, "claude-3-opus");
        assert_eq!(
            anthropic_req.system,
            Some("You are a helpful assistant".to_string())
        );
        assert_eq!(anthropic_req.messages.len(), 1);
        assert_eq!(anthropic_req.messages[0].role, "user");
    }
}

