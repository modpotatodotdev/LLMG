//! REST API handler for OpenAI Responses API
//!
//! Provides a JSON-based REST interface for the Responses API with SSE streaming,
//! complementing the WebSocket-based implementation.

use crate::response_store::{
    StoredIncompleteDetails, StoredReasoning, StoredResponse, StoredTextFormat,
    StoredTextFormatType, StoredUsage,
};
use crate::routing::parse_model_id;
use crate::GatewayState;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response, Sse},
    Json,
};
use futures::stream::{self, StreamExt};
use llmg_core::types::{ChatCompletionRequest, Message as ChatMessage, Tool};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Deserialize)]
pub struct CreateResponseRequest {
    pub model: String,
    #[serde(default)]
    pub input: ResponseInput,
    #[serde(rename = "previous_response_id", default)]
    pub previous_response_id: Option<String>,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub tools: Option<Vec<ResponseTool>>,
    #[serde(rename = "tool_choice", default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(rename = "parallel_tool_calls", default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(rename = "max_output_tokens", default)]
    pub max_output_tokens: Option<u32>,
    #[serde(rename = "top_p", default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub text: Option<ResponseTextConfig>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(rename = "stream_options", default)]
    pub stream_options: Option<ResponseStreamOptions>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    #[serde(default)]
    pub reasoning: Option<ReasoningConfig>,
    #[serde(rename = "service_tier", default)]
    pub service_tier: Option<String>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub store: Option<bool>,
    #[serde(default)]
    pub background: Option<bool>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ReasoningConfig {
    #[serde(default)]
    pub effort: Option<String>,
    #[serde(default)]
    pub summary: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ResponseInput {
    String(String),
    Array(Vec<serde_json::Value>),
}

impl Default for ResponseInput {
    fn default() -> Self {
        ResponseInput::String(String::new())
    }
}

impl ResponseInput {
    pub fn into_messages(self) -> Vec<ChatMessage> {
        match self {
            ResponseInput::String(s) => vec![ChatMessage::User {
                content: s,
                name: None,
            }],
            ResponseInput::Array(items) => convert_input_array(items),
        }
    }

    pub fn as_input_items(&self) -> Vec<serde_json::Value> {
        match self {
            ResponseInput::String(s) => vec![serde_json::json!({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": s}]
            })],
            ResponseInput::Array(items) => items.clone(),
        }
    }
}

fn convert_input_array(items: Vec<serde_json::Value>) -> Vec<ChatMessage> {
    let mut messages = Vec::new();

    for item in items {
        let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");

        match item_type {
            "message" => {
                let role = item.get("role").and_then(|v| v.as_str()).unwrap_or("user");
                let content = extract_content_from_item(&item);

                let msg = match role {
                    "system" => ChatMessage::System {
                        content,
                        name: None,
                    },
                    "user" => ChatMessage::User {
                        content,
                        name: None,
                    },
                    "assistant" => {
                        let tool_calls = extract_tool_calls_from_item(&item);
                        ChatMessage::Assistant {
                            content: Some(content),
                            refusal: None,
                            tool_calls,
                        }
                    }
                    _ => ChatMessage::User {
                        content,
                        name: None,
                    },
                };
                messages.push(msg);
            }
            "function_call_output" => {
                let output = item.get("output").and_then(|v| v.as_str()).unwrap_or("");
                let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                messages.push(ChatMessage::Tool {
                    content: output.to_string(),
                    tool_call_id: call_id.to_string(),
                });
            }
            "compaction" => {}
            _ => {}
        }
    }

    messages
}

fn extract_content_from_item(item: &serde_json::Value) -> String {
    let content_array = item.get("content").and_then(|v| v.as_array());

    if let Some(arr) = content_array {
        let mut parts = Vec::new();
        for c in arr {
            let block_type = c.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match block_type {
                "input_text" => {
                    if let Some(text) = c.get("text").and_then(|t| t.as_str()) {
                        parts.push(text.to_string());
                    }
                }
                "input_image" => {
                    if let Some(url) = c
                        .get("image_url")
                        .and_then(|v| v.get("url"))
                        .and_then(|v| v.as_str())
                    {
                        parts.push(url.to_string());
                    } else if let Some(data) = c.get("image_data").and_then(|v| v.as_str()) {
                        parts.push(data.to_string());
                    }
                }
                _ => {}
            }
        }
        parts.join("\n")
    } else {
        item.get("content")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string()
    }
}

fn extract_tool_calls_from_item(
    item: &serde_json::Value,
) -> Option<Vec<llmg_core::types::ToolCall>> {
    let content_array = item.get("content").and_then(|v| v.as_array())?;
    let mut tool_calls = Vec::new();

    for c in content_array {
        if c.get("type").and_then(|v| v.as_str()) == Some("function_call") {
            let call_id = c
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let name = c
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let arguments = c
                .get("arguments")
                .and_then(|v| v.as_str())
                .unwrap_or("{}")
                .to_string();

            tool_calls.push(llmg_core::types::ToolCall {
                id: call_id,
                r#type: "function".to_string(),
                function: llmg_core::types::FunctionCall { name, arguments },
            });
        }
    }

    if tool_calls.is_empty() {
        None
    } else {
        Some(tool_calls)
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct ResponseTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
    #[serde(default)]
    pub strict: Option<bool>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ResponseTextConfig {
    #[serde(default)]
    pub format: ResponseTextFormat,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseTextFormat {
    Text,
    JsonObject,
    #[serde(rename = "json_schema")]
    JsonSchema {
        name: String,
        schema: serde_json::Value,
        #[serde(default)]
        strict: Option<bool>,
    },
}

impl Default for ResponseTextFormat {
    fn default() -> Self {
        ResponseTextFormat::Text
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct ResponseStreamOptions {
    #[serde(rename = "include_usage", default)]
    pub include_usage: bool,
}

#[derive(Debug, Serialize)]
pub struct CreateResponseResponse {
    pub id: String,
    pub object: String,
    pub status: String,
    pub created_at: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<u64>,
    pub model: String,
    pub output: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ResponseError>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<ResponseIncompleteDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ResponseReasoningOutput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    pub text: ResponseTextOutput,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ResponseUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct ResponseError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ResponseIncompleteDetails {
    pub reason: String,
}

#[derive(Debug, Serialize)]
pub struct ResponseReasoningOutput {
    pub effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ResponseTextOutput {
    pub format: ResponseTextFormat,
}

#[derive(Debug, Serialize, Clone)]
pub struct ResponseUsage {
    pub input_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<ResponseInputTokensDetails>,
    pub output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<ResponseOutputTokensDetails>,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize, Clone)]
pub struct ResponseInputTokensDetails {
    pub cached_tokens: u32,
}

#[derive(Debug, Serialize, Clone)]
pub struct ResponseOutputTokensDetails {
    pub reasoning_tokens: u32,
}

#[derive(Debug, Deserialize)]
pub struct StreamQuery {
    #[serde(default)]
    pub stream: Option<bool>,
}

fn generate_response_id() -> String {
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
    let uuid = uuid::Uuid::new_v4();
    let bytes = uuid.as_bytes();
    format!("resp_{}", URL_SAFE_NO_PAD.encode(&bytes[..16]))
}

fn generate_message_id() -> String {
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
    let uuid = uuid::Uuid::new_v4();
    let bytes = uuid.as_bytes();
    format!("msg_{}", URL_SAFE_NO_PAD.encode(&bytes[..16]))
}

fn api_error(
    status: StatusCode,
    error_type: &str,
    code: &str,
    message: &str,
    param: Option<&str>,
) -> Response {
    (
        status,
        Json(serde_json::json!({
            "error": {
                "type": error_type,
                "code": code,
                "message": message,
                "param": param
            }
        })),
    )
        .into_response()
}

fn convert_tools(tools: Option<Vec<ResponseTool>>) -> Option<Vec<Tool>> {
    tools.map(|tools_list| {
        tools_list
            .into_iter()
            .filter_map(|t| {
                if t.tool_type == "function" {
                    Some(Tool {
                        r#type: "function".to_string(),
                        function: llmg_core::types::FunctionDefinition {
                            name: t.name,
                            description: t.description,
                            parameters: t.parameters.unwrap_or(serde_json::json!({})),
                        },
                    })
                } else {
                    None
                }
            })
            .collect()
    })
}

fn convert_tool_choice(tc: Option<serde_json::Value>) -> Option<llmg_core::types::ToolChoice> {
    tc.and_then(|tc| {
        if let Some(s) = tc.as_str() {
            Some(llmg_core::types::ToolChoice::String(s.to_string()))
        } else if let Some(obj) = tc.as_object() {
            obj.get("type")
                .and_then(|t| t.as_str())
                .and_then(|type_str| {
                    if type_str == "function" {
                        obj.get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(|n| n.as_str())
                            .map(|name| {
                                llmg_core::types::ToolChoice::Named(
                                    llmg_core::types::NamedToolChoice {
                                        r#type: "function".to_string(),
                                        function: llmg_core::types::FunctionName {
                                            name: name.to_string(),
                                        },
                                    },
                                )
                            })
                    } else {
                        None
                    }
                })
        } else {
            None
        }
    })
}

fn serialize_sse_event(event: &SseEvent) -> axum::response::sse::Event {
    let data = serde_json::to_string(event).unwrap_or_default();
    axum::response::sse::Event::default().data(data)
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SseEvent {
    #[serde(rename = "response.created")]
    ResponseCreated {
        response: SseResponseInfo,
    },
    #[serde(rename = "response.in_progress")]
    ResponseInProgress {
        response: SseResponseInfo,
    },
    #[serde(rename = "response.output_item.added")]
    ResponseOutputItemAdded {
        response_id: String,
        output_index: u32,
        item: serde_json::Value,
    },
    #[serde(rename = "response.output_item.done")]
    ResponseOutputItemDone {
        response_id: String,
        output_index: u32,
        item: serde_json::Value,
    },
    #[serde(rename = "response.content_part.added")]
    ResponseContentPartAdded {
        response_id: String,
        output_index: u32,
        content_index: u32,
        part: serde_json::Value,
    },
    #[serde(rename = "response.content_part.done")]
    ResponseContentPartDone {
        response_id: String,
        output_index: u32,
        content_index: u32,
        part: serde_json::Value,
    },
    #[serde(rename = "response.output_text.delta")]
    ResponseOutputTextDelta {
        response_id: String,
        output_index: u32,
        content_index: u32,
        delta: String,
    },
    #[serde(rename = "response.output_text.done")]
    ResponseOutputTextDone {
        response_id: String,
        output_index: u32,
        content_index: u32,
        text: String,
    },
    #[serde(rename = "response.function_call_arguments.delta")]
    ResponseFunctionCallArgumentsDelta {
        response_id: String,
        output_index: u32,
        call_id: String,
        delta: String,
    },
    #[serde(rename = "response.function_call_arguments.done")]
    ResponseFunctionCallArgumentsDone {
        response_id: String,
        output_index: u32,
        call_id: String,
        arguments: String,
    },
    #[serde(rename = "response.done")]
    ResponseDone {
        response: CreateResponseResponse,
    },
    #[serde(rename = "response.failed")]
    ResponseFailed {
        response_id: String,
        error: ResponseError,
    },
    Error {
        error: ResponseError,
    },
}

#[derive(Debug, Serialize)]
struct SseResponseInfo {
    id: String,
    object: String,
    status: String,
    created_at: u64,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_response_id: Option<String>,
}

pub async fn create_response_handler(
    State(state): State<Arc<GatewayState>>,
    Query(query): Query<StreamQuery>,
    Json(request): Json<CreateResponseRequest>,
) -> Response {
    let should_stream = request.stream.unwrap_or(false) || query.stream.unwrap_or(false);

    if should_stream {
        return handle_streaming_response(state, request).await;
    }

    handle_non_streaming_response(state, request).await
}

async fn handle_non_streaming_response(
    state: Arc<GatewayState>,
    request: CreateResponseRequest,
) -> Response {
    let response_id = generate_response_id();
    let message_id = generate_message_id();
    let created_at = chrono::Utc::now().timestamp() as u64;

    let resolved_model = state
        .config
        .aliases
        .get(&request.model)
        .cloned()
        .unwrap_or_else(|| request.model.clone());

    let (provider_name, model_name) = match parse_model_id(&resolved_model) {
        Ok((p, m)) => (p.to_string(), m),
        Err(e) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "invalid_value",
                &format!("Invalid model format: {:?}", e),
                Some("model"),
            );
        }
    };

    let provider = match state.registry.get(&provider_name) {
        Some(p) => p.clone(),
        None => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "not_found",
                &format!("Unknown provider: {}", provider_name),
                None,
            );
        }
    };

    let mut messages = Vec::new();
    if let Some(ref instructions) = request.instructions {
        messages.push(ChatMessage::System {
            content: instructions.clone(),
            name: None,
        });
    }

    let input_items = request.input.as_input_items();
    messages.extend(request.input.into_messages());

    let raw_tools = request.tools.clone();
    let raw_tool_choice = request.tool_choice.clone();
    let tools = convert_tools(request.tools);
    let tool_choice = convert_tool_choice(request.tool_choice);

    let chat_request = ChatCompletionRequest {
        model: model_name,
        messages,
        stream: Some(false),
        temperature: request.temperature,
        max_tokens: request.max_output_tokens,
        top_p: request.top_p,
        frequency_penalty: None,
        presence_penalty: None,
        stop: None,
        user: None,
        tools,
        tool_choice,
    };

    match provider.chat_completion(chat_request).await {
        Ok(response) => {
            let completed_at = chrono::Utc::now().timestamp() as u64;

            let output_text = response
                .choices
                .first()
                .and_then(|c| match &c.message {
                    ChatMessage::Assistant { content, .. } => content.clone(),
                    _ => None,
                })
                .unwrap_or_default();

            let usage = response.usage.map(|u| ResponseUsage {
                input_tokens: u.prompt_tokens,
                input_tokens_details: None,
                output_tokens: u.completion_tokens,
                output_tokens_details: None,
                total_tokens: u.total_tokens,
            });

            let text_format = request.text.map(|t| t.format).unwrap_or_default();

            let serialized_tools = raw_tools.as_ref().map(|t| {
                t.iter()
                    .filter_map(|tool| {
                        if tool.tool_type == "function" {
                            Some(serde_json::json!({
                                "type": "function",
                                "function": {
                                    "name": tool.name,
                                    "description": tool.description,
                                    "parameters": tool.parameters
                                }
                            }))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            });

            let response_obj = CreateResponseResponse {
                id: response_id.clone(),
                object: "response".to_string(),
                status: "completed".to_string(),
                created_at,
                completed_at: Some(completed_at),
                model: request.model.clone(),
                output: vec![serde_json::json!({
                    "type": "message",
                    "id": message_id,
                    "status": "completed",
                    "role": "assistant",
                    "content": [{
                        "type": "output_text",
                        "text": output_text,
                        "annotations": []
                    }]
                })],
                error: None,
                incomplete_details: None,
                instructions: request.instructions.clone(),
                max_output_tokens: request.max_output_tokens,
                previous_response_id: request.previous_response_id.clone(),
                reasoning: request.reasoning.as_ref().map(|r| ResponseReasoningOutput {
                    effort: r.effort.clone(),
                    summary: r.summary.clone(),
                }),
                store: request.store,
                service_tier: request.service_tier.clone(),
                temperature: request.temperature,
                text: ResponseTextOutput {
                    format: text_format,
                },
                tool_choice: raw_tool_choice.clone(),
                tools: serialized_tools.clone(),
                top_p: request.top_p,
                truncation: None,
                usage,
                metadata: request.metadata.clone(),
                user: request.user.clone(),
                parallel_tool_calls: request.parallel_tool_calls,
            };

            if request.store.unwrap_or(true) {
                let stored = StoredResponse {
                    id: response_id,
                    object: "response".to_string(),
                    status: "completed".to_string(),
                    created_at,
                    completed_at: Some(completed_at),
                    model: request.model,
                    output: response_obj.output.clone(),
                    error: None,
                    incomplete_details: None,
                    instructions: request.instructions,
                    max_output_tokens: request.max_output_tokens,
                    previous_response_id: request.previous_response_id,
                    reasoning: request.reasoning.map(|r| StoredReasoning {
                        effort: r.effort,
                        summary: r.summary,
                    }),
                    store: true,
                    service_tier: request.service_tier,
                    temperature: request.temperature,
                    text: StoredTextFormat {
                        format: StoredTextFormatType::Text,
                    },
                    tool_choice: raw_tool_choice,
                    tools: serialized_tools,
                    top_p: request.top_p,
                    truncation: None,
                    usage: response_obj.usage.as_ref().map(|u| StoredUsage {
                        input_tokens: u.input_tokens,
                        input_tokens_details: None,
                        output_tokens: u.output_tokens,
                        output_tokens_details: None,
                        total_tokens: u.total_tokens,
                    }),
                    metadata: request.metadata,
                    user: request.user,
                    parallel_tool_calls: request.parallel_tool_calls,
                    input_items,
                };
                state.response_store.store(stored).await;
            }

            (StatusCode::OK, Json(response_obj)).into_response()
        }
        Err(e) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            "internal_error",
            &e.to_string(),
            None,
        ),
    }
}

async fn handle_streaming_response(
    state: Arc<GatewayState>,
    request: CreateResponseRequest,
) -> Response {
    let response_id = generate_response_id();
    let created_at = chrono::Utc::now().timestamp() as u64;

    let resolved_model = state
        .config
        .aliases
        .get(&request.model)
        .cloned()
        .unwrap_or_else(|| request.model.clone());

    let (provider_name, model_name) = match parse_model_id(&resolved_model) {
        Ok((p, m)) => (p.to_string(), m),
        Err(e) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "invalid_value",
                &format!("Invalid model format: {:?}", e),
                Some("model"),
            );
        }
    };

    let provider = match state.registry.get(&provider_name) {
        Some(p) => p.clone(),
        None => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "not_found",
                &format!("Unknown provider: {}", provider_name),
                None,
            );
        }
    };

    let mut messages = Vec::new();
    if let Some(ref instructions) = request.instructions {
        messages.push(ChatMessage::System {
            content: instructions.clone(),
            name: None,
        });
    }
    messages.extend(request.input.into_messages());

    let tools = convert_tools(request.tools.clone());
    let tool_choice = convert_tool_choice(request.tool_choice.clone());

    let chat_request = ChatCompletionRequest {
        model: model_name,
        messages,
        stream: Some(true),
        temperature: request.temperature,
        max_tokens: request.max_output_tokens,
        top_p: request.top_p,
        frequency_penalty: None,
        presence_penalty: None,
        stop: None,
        user: None,
        tools,
        tool_choice,
    };

    let stream = match provider.chat_completion_stream(chat_request).await {
        Ok(s) => s,
        Err(e) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "internal_error",
                &e.to_string(),
                None,
            );
        }
    };

    let response_id_clone = response_id.clone();
    let model_clone = request.model.clone();
    let instructions_clone = request.instructions.clone();
    let previous_response_id = request.previous_response_id.clone();
    let store = request.store;
    let service_tier = request.service_tier;
    let temperature = request.temperature;
    let max_output_tokens = request.max_output_tokens;
    let top_p = request.top_p;
    let tool_choice_serialized = request.tool_choice.clone();
    let tools_serialized = request.tools.clone();
    let metadata = request.metadata.clone();
    let user = request.user.clone();
    let parallel_tool_calls = request.parallel_tool_calls;
    let reasoning = request.reasoning.clone();
    let text_format = request.text.map(|t| t.format).unwrap_or_default();
    let include_usage = request
        .stream_options
        .map(|o| o.include_usage)
        .unwrap_or(false);

    let response_store = state.response_store.clone();

    let sse_stream = stream::unfold(
        (
            stream,
            false,
            String::new(),
            0u32,
            Vec::<serde_json::Value>::new(),
            0u32,
            0u32,
            false,
        ),
        move |(
            mut stream,
            mut item_added,
            mut full_text,
            mut output_index,
            mut output_items,
            mut usage_input,
            mut usage_output,
            mut done,
        )| {
            let rid = response_id_clone.clone();
            let mid = generate_message_id();
            let model = model_clone.clone();
            let instr = instructions_clone.clone();
            let prev_id = previous_response_id.clone();
            let txt_fmt = text_format.clone();
            let reasoning_clone = reasoning.clone();
            let service_tier_clone = service_tier.clone();
            let tool_choice_clone = tool_choice_serialized.clone();
            let tools_clone = tools_serialized.clone();
            let metadata_clone = metadata.clone();
            let user_clone = user.clone();
            let response_store_clone = response_store.clone();

            async move {
                if done {
                    return None;
                }

                use futures::StreamExt;
                match stream.next().await {
                    Some(Ok(chunk)) => {
                        if include_usage {
                            if let Some(usage) = &chunk.usage {
                                usage_input = usage.prompt_tokens;
                                usage_output = usage.completion_tokens;
                            }
                        }

                        let mut events: Vec<
                            Result<axum::response::sse::Event, std::convert::Infallible>,
                        > = Vec::new();

                        if !item_added {
                            item_added = true;
                            events.push(Ok(serialize_sse_event(&SseEvent::ResponseCreated {
                                response: SseResponseInfo {
                                    id: rid.clone(),
                                    object: "response".to_string(),
                                    status: "in_progress".to_string(),
                                    created_at,
                                    model: model.clone(),
                                    previous_response_id: prev_id.clone(),
                                },
                            })));
                            events.push(Ok(serialize_sse_event(&SseEvent::ResponseInProgress {
                                response: SseResponseInfo {
                                    id: rid.clone(),
                                    object: "response".to_string(),
                                    status: "in_progress".to_string(),
                                    created_at,
                                    model: model.clone(),
                                    previous_response_id: prev_id.clone(),
                                },
                            })));
                            events.push(Ok(serialize_sse_event(
                                &SseEvent::ResponseOutputItemAdded {
                                    response_id: rid.clone(),
                                    output_index: 0,
                                    item: serde_json::json!({
                                        "type": "message",
                                        "id": mid.clone(),
                                        "status": "in_progress",
                                        "role": "assistant",
                                        "content": []
                                    }),
                                },
                            )));
                            events.push(Ok(serialize_sse_event(
                                &SseEvent::ResponseContentPartAdded {
                                    response_id: rid.clone(),
                                    output_index: 0,
                                    content_index: 0,
                                    part: serde_json::json!({
                                        "type": "output_text",
                                        "text": "",
                                        "annotations": []
                                    }),
                                },
                            )));
                        }

                        for choice in &chunk.choices {
                            if let Some(content) = &choice.delta.content {
                                full_text.push_str(content);
                                events.push(Ok(serialize_sse_event(
                                    &SseEvent::ResponseOutputTextDelta {
                                        response_id: rid.clone(),
                                        output_index: 0,
                                        content_index: 0,
                                        delta: content.clone(),
                                    },
                                )));
                            }

                            if let Some(tc_deltas) = &choice.delta.tool_calls {
                                for tc_delta in tc_deltas {
                                    let call_id = tc_delta.id.clone().unwrap_or_else(|| {
                                        format!("call_{}", uuid::Uuid::new_v4())
                                    });
                                    let args = tc_delta
                                        .function
                                        .as_ref()
                                        .and_then(|f| f.arguments.clone())
                                        .unwrap_or_default();
                                    events.push(Ok(serialize_sse_event(
                                        &SseEvent::ResponseFunctionCallArgumentsDelta {
                                            response_id: rid.clone(),
                                            output_index: output_index + 1,
                                            call_id: call_id.clone(),
                                            delta: args,
                                        },
                                    )));
                                }
                            }
                        }

                        Some((
                            events,
                            (
                                stream,
                                item_added,
                                full_text,
                                output_index,
                                output_items,
                                usage_input,
                                usage_output,
                                done,
                            ),
                        ))
                    }
                    Some(Err(e)) => {
                        let events: Vec<
                            Result<axum::response::sse::Event, std::convert::Infallible>,
                        > = vec![Ok(serialize_sse_event(&SseEvent::Error {
                            error: ResponseError {
                                error_type: "server_error".to_string(),
                                code: "stream_error".to_string(),
                                message: e.to_string(),
                                param: None,
                            },
                        }))];
                        Some((
                            events,
                            (
                                stream,
                                item_added,
                                full_text,
                                output_index,
                                output_items,
                                usage_input,
                                usage_output,
                                true,
                            ),
                        ))
                    }
                    None => {
                        let completed_at = chrono::Utc::now().timestamp() as u64;
                        let mut events: Vec<
                            Result<axum::response::sse::Event, std::convert::Infallible>,
                        > = Vec::new();

                        events.push(Ok(serialize_sse_event(&SseEvent::ResponseOutputTextDone {
                            response_id: rid.clone(),
                            output_index: 0,
                            content_index: 0,
                            text: full_text.clone(),
                        })));
                        events.push(Ok(serialize_sse_event(
                            &SseEvent::ResponseContentPartDone {
                                response_id: rid.clone(),
                                output_index: 0,
                                content_index: 0,
                                part: serde_json::json!({
                                    "type": "output_text",
                                    "text": full_text,
                                    "annotations": []
                                }),
                            },
                        )));

                        let output_item = serde_json::json!({
                            "type": "message",
                            "id": mid,
                            "status": "completed",
                            "role": "assistant",
                            "content": [{
                                "type": "output_text",
                                "text": full_text,
                                "annotations": []
                            }]
                        });
                        events.push(Ok(serialize_sse_event(&SseEvent::ResponseOutputItemDone {
                            response_id: rid.clone(),
                            output_index: 0,
                            item: output_item.clone(),
                        })));

                        let usage = if include_usage && (usage_input > 0 || usage_output > 0) {
                            Some(ResponseUsage {
                                input_tokens: usage_input,
                                input_tokens_details: None,
                                output_tokens: usage_output,
                                output_tokens_details: None,
                                total_tokens: usage_input.saturating_add(usage_output),
                            })
                        } else {
                            None
                        };

                        let done_response = CreateResponseResponse {
                            id: rid.clone(),
                            object: "response".to_string(),
                            status: "completed".to_string(),
                            created_at,
                            completed_at: Some(completed_at),
                            model: model.clone(),
                            output: vec![output_item],
                            error: None,
                            incomplete_details: None,
                            instructions: instr.clone(),
                            max_output_tokens,
                            previous_response_id: prev_id.clone(),
                            reasoning: reasoning_clone.as_ref().map(|r| ResponseReasoningOutput {
                                effort: r.effort.clone(),
                                summary: r.summary.clone(),
                            }),
                            store,
                            service_tier: service_tier_clone.clone(),
                            temperature,
                            text: ResponseTextOutput { format: txt_fmt },
                            tool_choice: tool_choice_clone.clone(),
                            tools: tools_clone.as_ref().map(|t| {
                                t.iter()
                                    .filter_map(|tool: &ResponseTool| {
                                        if tool.tool_type == "function" {
                                            Some(serde_json::json!({
                                                "type": "function",
                                                "function": {
                                                    "name": tool.name,
                                                    "description": tool.description,
                                                    "parameters": tool.parameters
                                                }
                                            }))
                                        } else {
                                            None
                                        }
                                    })
                                    .collect()
                            }),
                            top_p,
                            truncation: None,
                            usage: usage.clone(),
                            metadata: metadata_clone.clone(),
                            user: user_clone.clone(),
                            parallel_tool_calls,
                        };

                        events.push(Ok(serialize_sse_event(&SseEvent::ResponseDone {
                            response: done_response,
                        })));

                        if store.unwrap_or(true) {
                            let stored = StoredResponse {
                                id: rid.clone(),
                                object: "response".to_string(),
                                status: "completed".to_string(),
                                created_at,
                                completed_at: Some(completed_at),
                                model: model.clone(),
                                output: vec![serde_json::json!({
                                    "type": "message",
                                    "id": mid,
                                    "status": "completed",
                                    "role": "assistant",
                                    "content": [{
                                        "type": "output_text",
                                        "text": full_text,
                                        "annotations": []
                                    }]
                                })],
                                error: None,
                                incomplete_details: None,
                                instructions: instr,
                                max_output_tokens,
                                previous_response_id: prev_id,
                                reasoning: reasoning_clone.map(|r| StoredReasoning {
                                    effort: r.effort,
                                    summary: r.summary,
                                }),
                                store: true,
                                service_tier: service_tier_clone,
                                temperature,
                                text: StoredTextFormat {
                                    format: StoredTextFormatType::Text,
                                },
                                tool_choice: tool_choice_clone,
                                tools: tools_clone.map(|t| {
                                    t.into_iter()
                                        .filter_map(|tool: ResponseTool| {
                                            if tool.tool_type == "function" {
                                                Some(serde_json::json!({
                                                    "type": "function",
                                                    "function": {
                                                        "name": tool.name,
                                                        "description": tool.description,
                                                        "parameters": tool.parameters
                                                    }
                                                }))
                                            } else {
                                                None
                                            }
                                        })
                                        .collect()
                                }),
                                top_p,
                                truncation: None,
                                usage: usage.map(|u| StoredUsage {
                                    input_tokens: u.input_tokens,
                                    input_tokens_details: None,
                                    output_tokens: u.output_tokens,
                                    output_tokens_details: None,
                                    total_tokens: u.total_tokens,
                                }),
                                metadata: metadata_clone,
                                user: user_clone,
                                parallel_tool_calls,
                                input_items: vec![],
                            };
                            response_store_clone.store(stored).await;
                        }

                        Some((
                            events,
                            (
                                stream,
                                item_added,
                                full_text,
                                output_index,
                                output_items,
                                usage_input,
                                usage_output,
                                true,
                            ),
                        ))
                    }
                }
            }
        },
    )
    .flat_map(stream::iter);

    Sse::new(sse_stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

pub async fn retrieve_response_handler(
    State(state): State<Arc<GatewayState>>,
    Path(response_id): Path<String>,
) -> Response {
    match state.response_store.get(&response_id).await {
        Some(stored) => {
            let response = stored_to_response(stored);
            (StatusCode::OK, Json(response)).into_response()
        }
        None => api_error(
            StatusCode::NOT_FOUND,
            "invalid_request_error",
            "not_found",
            &format!("Response '{}' not found", response_id),
            None,
        ),
    }
}

pub async fn delete_response_handler(
    State(state): State<Arc<GatewayState>>,
    Path(response_id): Path<String>,
) -> Response {
    if state.response_store.delete(&response_id).await {
        (StatusCode::OK, Json(serde_json::json!({ "deleted": true }))).into_response()
    } else {
        api_error(
            StatusCode::NOT_FOUND,
            "invalid_request_error",
            "not_found",
            &format!("Response '{}' not found", response_id),
            None,
        )
    }
}

pub async fn cancel_response_handler(
    State(state): State<Arc<GatewayState>>,
    Path(response_id): Path<String>,
) -> Response {
    match state.response_store.get(&response_id).await {
        Some(mut stored) => {
            if stored.status == "completed" {
                return api_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "already_completed",
                    "Response is already completed",
                    None,
                );
            }

            stored.status = "cancelled".to_string();
            stored.incomplete_details = Some(StoredIncompleteDetails {
                reason: "cancelled".to_string(),
            });
            stored.completed_at = Some(chrono::Utc::now().timestamp() as u64);
            state.response_store.store(stored.clone()).await;

            let response = stored_to_response(stored);
            (StatusCode::OK, Json(response)).into_response()
        }
        None => api_error(
            StatusCode::NOT_FOUND,
            "invalid_request_error",
            "not_found",
            &format!("Response '{}' not found", response_id),
            None,
        ),
    }
}

pub async fn list_input_items_handler(
    State(state): State<Arc<GatewayState>>,
    Path(response_id): Path<String>,
) -> Response {
    match state.response_store.get(&response_id).await {
        Some(stored) => {
            let items = stored.input_items;
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "object": "list",
                    "data": items,
                    "has_more": false
                })),
            )
                .into_response()
        }
        None => api_error(
            StatusCode::NOT_FOUND,
            "invalid_request_error",
            "not_found",
            &format!("Response '{}' not found", response_id),
            None,
        ),
    }
}

pub async fn count_tokens_handler(
    State(state): State<Arc<GatewayState>>,
    Json(request): Json<CreateResponseRequest>,
) -> Response {
    let mut messages = Vec::new();
    if let Some(ref instructions) = request.instructions {
        messages.push(ChatMessage::System {
            content: instructions.clone(),
            name: None,
        });
    }
    messages.extend(request.input.into_messages());

    let text = messages
        .iter()
        .map(|m| match m {
            ChatMessage::System { content, .. } => content.clone(),
            ChatMessage::User { content, .. } => content.clone(),
            ChatMessage::Assistant { content, .. } => content.clone().unwrap_or_default(),
            ChatMessage::Tool { content, .. } => content.clone(),
        })
        .collect::<Vec<_>>()
        .join("\n");

    let token_estimate = (text.len() as f64 / 4.0).ceil() as u32;

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "input_tokens": token_estimate
        })),
    )
        .into_response()
}

#[derive(Debug, Deserialize)]
pub struct SubmitToolOutputsRequest {
    pub tool_outputs: Vec<ToolOutput>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(rename = "stream_options", default)]
    pub stream_options: Option<ResponseStreamOptions>,
}

#[derive(Debug, Deserialize)]
pub struct ToolOutput {
    #[serde(rename = "tool_call_id")]
    pub tool_call_id: String,
    pub output: String,
}

pub async fn submit_tool_outputs_handler(
    State(state): State<Arc<GatewayState>>,
    Path(response_id): Path<String>,
    Query(query): Query<StreamQuery>,
    Json(request): Json<SubmitToolOutputsRequest>,
) -> Response {
    let should_stream = request.stream.unwrap_or(false) || query.stream.unwrap_or(false);

    // Retrieve the original response
    let stored_response = match state.response_store.get(&response_id).await {
        Some(r) => r,
        None => {
            return api_error(
                StatusCode::NOT_FOUND,
                "invalid_request_error",
                "not_found",
                &format!("Response '{}' not found", response_id),
                None,
            );
        }
    };

    // Get the provider
    let (provider_name, model_name) = match parse_model_id(&stored_response.model) {
        Ok((p, m)) => (p.to_string(), m),
        Err(e) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "invalid_value",
                &format!("Invalid model format: {:?}", e),
                Some("model"),
            );
        }
    };

    let provider = match state.registry.get(&provider_name) {
        Some(p) => p.clone(),
        None => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "not_found",
                &format!("Unknown provider: {}", provider_name),
                None,
            );
        }
    };

    // Build conversation history from the stored response
    let mut messages = Vec::new();

    // Add instructions if present
    if let Some(ref instructions) = stored_response.instructions {
        messages.push(ChatMessage::System {
            content: instructions.clone(),
            name: None,
        });
    }

    // Add the original input items as messages
    let input_items = stored_response.input_items.clone();
    for item in &input_items {
        if let Some(item_type) = item.get("type").and_then(|v| v.as_str()) {
            match item_type {
                "message" => {
                    if let Some(role) = item.get("role").and_then(|v| v.as_str()) {
                        let content = extract_content_from_item(item);
                        let msg = match role {
                            "system" => ChatMessage::System {
                                content,
                                name: None,
                            },
                            "user" => ChatMessage::User {
                                content,
                                name: None,
                            },
                            "assistant" => {
                                let tool_calls = extract_tool_calls_from_item(item);
                                ChatMessage::Assistant {
                                    content: Some(content),
                                    refusal: None,
                                    tool_calls,
                                }
                            }
                            _ => ChatMessage::User {
                                content,
                                name: None,
                            },
                        };
                        messages.push(msg);
                    }
                }
                "function_call_output" => {
                    if let (Some(output), Some(call_id)) = (
                        item.get("output").and_then(|v| v.as_str()),
                        item.get("call_id").and_then(|v| v.as_str()),
                    ) {
                        messages.push(ChatMessage::Tool {
                            content: output.to_string(),
                            tool_call_id: call_id.to_string(),
                        });
                    }
                }
                _ => {}
            }
        }
    }

    // Add the assistant's function call from the stored response output
    for output_item in &stored_response.output {
        if let Some(item_type) = output_item.get("type").and_then(|v| v.as_str()) {
            if item_type == "function_call" {
                if let (Some(call_id), Some(name), Some(arguments)) = (
                    output_item.get("id").and_then(|v| v.as_str()),
                    output_item.get("name").and_then(|v| v.as_str()),
                    output_item.get("arguments").and_then(|v| v.as_str()),
                ) {
                    messages.push(ChatMessage::Assistant {
                        content: None,
                        refusal: None,
                        tool_calls: Some(vec![llmg_core::types::ToolCall {
                            id: call_id.to_string(),
                            r#type: "function".to_string(),
                            function: llmg_core::types::FunctionCall {
                                name: name.to_string(),
                                arguments: arguments.to_string(),
                            },
                        }]),
                    });
                }
            }
        }
    }

    // Add the tool outputs as function_call_output items
    for tool_output in &request.tool_outputs {
        messages.push(ChatMessage::Tool {
            content: tool_output.output.clone(),
            tool_call_id: tool_output.tool_call_id.clone(),
        });
    }

    // Create the chat completion request
    let chat_request = ChatCompletionRequest {
        model: model_name,
        messages,
        stream: Some(should_stream),
        temperature: stored_response.temperature,
        max_tokens: stored_response.max_output_tokens,
        top_p: stored_response.top_p,
        frequency_penalty: None,
        presence_penalty: None,
        stop: None,
        user: stored_response.user.clone(),
        tools: stored_response.tools.as_ref().map(|tools| {
            tools
                .iter()
                .filter_map(|t| {
                    if let Some(func) = t.get("function") {
                        Some(llmg_core::types::Tool {
                            r#type: "function".to_string(),
                            function: llmg_core::types::FunctionDefinition {
                                name: func.get("name")?.as_str()?.to_string(),
                                description: func
                                    .get("description")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string()),
                                parameters: func
                                    .get("parameters")
                                    .cloned()
                                    .unwrap_or(serde_json::json!({})),
                            },
                        })
                    } else {
                        None
                    }
                })
                .collect()
        }),
        tool_choice: stored_response.tool_choice.as_ref().map(|tc| {
            if let Some(s) = tc.as_str() {
                llmg_core::types::ToolChoice::String(s.to_string())
            } else {
                llmg_core::types::ToolChoice::String("auto".to_string())
            }
        }),
    };

    if should_stream {
        handle_streaming_submit_tool_outputs(state, chat_request, stored_response, request).await
    } else {
        handle_non_streaming_submit_tool_outputs(state, chat_request, stored_response, request)
            .await
    }
}

async fn handle_non_streaming_submit_tool_outputs(
    state: Arc<GatewayState>,
    chat_request: ChatCompletionRequest,
    stored_response: StoredResponse,
    _request: SubmitToolOutputsRequest,
) -> Response {
    let provider_name = match parse_model_id(&stored_response.model) {
        Ok((p, _)) => p.to_string(),
        Err(_) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "invalid_value",
                "Invalid model format",
                Some("model"),
            );
        }
    };

    let provider = match state.registry.get(&provider_name) {
        Some(p) => p.clone(),
        None => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "not_found",
                &format!("Unknown provider: {}", provider_name),
                None,
            );
        }
    };

    match provider.chat_completion(chat_request).await {
        Ok(response) => {
            let completed_at = chrono::Utc::now().timestamp() as u64;
            let new_response_id = generate_response_id();
            let message_id = generate_message_id();

            let output_text = response
                .choices
                .first()
                .and_then(|c| match &c.message {
                    ChatMessage::Assistant { content, .. } => content.clone(),
                    _ => None,
                })
                .unwrap_or_default();

            let usage = response.usage.map(|u| ResponseUsage {
                input_tokens: u.prompt_tokens,
                input_tokens_details: None,
                output_tokens: u.completion_tokens,
                output_tokens_details: None,
                total_tokens: u.total_tokens,
            });

            let response_obj = CreateResponseResponse {
                id: new_response_id.clone(),
                object: "response".to_string(),
                status: "completed".to_string(),
                created_at: chrono::Utc::now().timestamp() as u64,
                completed_at: Some(completed_at),
                model: stored_response.model.clone(),
                output: vec![serde_json::json!({
                    "type": "message",
                    "id": message_id,
                    "status": "completed",
                    "role": "assistant",
                    "content": [{
                        "type": "output_text",
                        "text": output_text,
                        "annotations": []
                    }]
                })],
                error: None,
                incomplete_details: None,
                instructions: stored_response.instructions.clone(),
                max_output_tokens: stored_response.max_output_tokens,
                previous_response_id: Some(stored_response.id.clone()),
                reasoning: stored_response
                    .reasoning
                    .as_ref()
                    .map(|r| ResponseReasoningOutput {
                        effort: r.effort.clone(),
                        summary: r.summary.clone(),
                    }),
                store: Some(stored_response.store),
                service_tier: stored_response.service_tier.clone(),
                temperature: stored_response.temperature,
                text: ResponseTextOutput {
                    format: ResponseTextFormat::Text,
                },
                tool_choice: stored_response.tool_choice.clone(),
                tools: stored_response.tools.clone(),
                top_p: stored_response.top_p,
                truncation: stored_response.truncation.clone(),
                usage,
                metadata: stored_response.metadata.clone(),
                user: stored_response.user.clone(),
                parallel_tool_calls: stored_response.parallel_tool_calls,
            };

            // Store the new response
            if stored_response.store {
                let stored = StoredResponse {
                    id: new_response_id.clone(),
                    object: "response".to_string(),
                    status: "completed".to_string(),
                    created_at: response_obj.created_at,
                    completed_at: response_obj.completed_at,
                    model: stored_response.model.clone(),
                    output: response_obj.output.clone(),
                    error: None,
                    incomplete_details: None,
                    instructions: stored_response.instructions.clone(),
                    max_output_tokens: stored_response.max_output_tokens,
                    previous_response_id: Some(stored_response.id.clone()),
                    reasoning: stored_response.reasoning.clone(),
                    store: stored_response.store,
                    service_tier: stored_response.service_tier.clone(),
                    temperature: stored_response.temperature,
                    text: StoredTextFormat {
                        format: StoredTextFormatType::Text,
                    },
                    tool_choice: stored_response.tool_choice.clone(),
                    tools: stored_response.tools.clone(),
                    top_p: stored_response.top_p,
                    truncation: stored_response.truncation.clone(),
                    usage: response_obj.usage.as_ref().map(|u| StoredUsage {
                        input_tokens: u.input_tokens,
                        input_tokens_details: None,
                        output_tokens: u.output_tokens,
                        output_tokens_details: None,
                        total_tokens: u.total_tokens,
                    }),
                    metadata: stored_response.metadata.clone(),
                    user: stored_response.user.clone(),
                    parallel_tool_calls: stored_response.parallel_tool_calls,
                    input_items: stored_response.input_items.clone(),
                };
                state.response_store.store(stored).await;
            }

            (StatusCode::OK, Json(response_obj)).into_response()
        }
        Err(e) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            "internal_error",
            &e.to_string(),
            None,
        ),
    }
}

async fn handle_streaming_submit_tool_outputs(
    _state: Arc<GatewayState>,
    _chat_request: ChatCompletionRequest,
    _stored_response: StoredResponse,
    _request: SubmitToolOutputsRequest,
) -> Response {
    // For now, return not implemented for streaming
    // TODO: Implement streaming support similar to handle_streaming_response
    api_error(
        StatusCode::NOT_IMPLEMENTED,
        "invalid_request_error",
        "not_implemented",
        "Streaming submit_tool_outputs is not yet implemented. Use stream: false for now.",
        None,
    )
}

pub async fn compact_response_handler(
    State(_state): State<Arc<GatewayState>>,
    Path(response_id): Path<String>,
) -> Response {
    api_error(
        StatusCode::NOT_IMPLEMENTED,
        "invalid_request_error",
        "not_implemented",
        "Response compaction requires LLM summarization and is not yet supported",
        None,
    )
}

fn stored_to_response(stored: StoredResponse) -> CreateResponseResponse {
    CreateResponseResponse {
        id: stored.id,
        object: stored.object,
        status: stored.status,
        created_at: stored.created_at,
        completed_at: stored.completed_at,
        model: stored.model,
        output: stored.output,
        error: stored.error.map(|e| ResponseError {
            error_type: "invalid_request_error".to_string(),
            code: e.code,
            message: e.message,
            param: e.param,
        }),
        incomplete_details: stored
            .incomplete_details
            .map(|d| ResponseIncompleteDetails { reason: d.reason }),
        instructions: stored.instructions,
        max_output_tokens: stored.max_output_tokens,
        previous_response_id: stored.previous_response_id,
        reasoning: stored.reasoning.map(|r| ResponseReasoningOutput {
            effort: r.effort,
            summary: r.summary,
        }),
        store: Some(stored.store),
        service_tier: stored.service_tier,
        temperature: stored.temperature,
        text: ResponseTextOutput {
            format: match stored.text.format {
                StoredTextFormatType::Text => ResponseTextFormat::Text,
                StoredTextFormatType::JsonObject => ResponseTextFormat::JsonObject,
                StoredTextFormatType::JsonSchema { name, schema } => {
                    ResponseTextFormat::JsonSchema {
                        name,
                        schema,
                        strict: None,
                    }
                }
            },
        },
        tool_choice: stored.tool_choice,
        tools: stored.tools,
        top_p: stored.top_p,
        truncation: stored.truncation,
        usage: stored.usage.map(|u| ResponseUsage {
            input_tokens: u.input_tokens,
            input_tokens_details: u.input_tokens_details.map(|d| ResponseInputTokensDetails {
                cached_tokens: d.cached_tokens,
            }),
            output_tokens: u.output_tokens,
            output_tokens_details: u
                .output_tokens_details
                .map(|d| ResponseOutputTokensDetails {
                    reasoning_tokens: d.reasoning_tokens,
                }),
            total_tokens: u.total_tokens,
        }),
        metadata: stored.metadata,
        user: stored.user,
        parallel_tool_calls: stored.parallel_tool_calls,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_create_response_request() {
        let json = r#"{
            "model": "openai/gpt-4",
            "input": "Hello",
            "instructions": "Be helpful",
            "temperature": 0.7,
            "max_output_tokens": 100,
            "top_p": 0.9,
            "stream": false,
            "store": true,
            "reasoning": {"effort": "high"},
            "service_tier": "auto"
        }"#;

        let request: CreateResponseRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.model, "openai/gpt-4");
        assert_eq!(request.temperature, Some(0.7));
        assert_eq!(request.max_output_tokens, Some(100));
        assert_eq!(request.top_p, Some(0.9));
        assert_eq!(request.stream, Some(false));
        assert_eq!(request.store, Some(true));
        assert_eq!(
            request.reasoning.as_ref().unwrap().effort,
            Some("high".to_string())
        );
        assert_eq!(request.service_tier, Some("auto".to_string()));
    }

    #[test]
    fn test_parse_input_string() {
        let json = r#"{
            "model": "gpt-4",
            "input": "Hello world"
        }"#;

        let request: CreateResponseRequest = serde_json::from_str(json).unwrap();
        match request.input {
            ResponseInput::String(s) => assert_eq!(s, "Hello world"),
            _ => panic!("Expected string input"),
        }
    }

    #[test]
    fn test_parse_input_array() {
        let json = r#"{
            "model": "gpt-4",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
                {"type": "function_call_output", "call_id": "call_123", "output": "result"}
            ]
        }"#;

        let request: CreateResponseRequest = serde_json::from_str(json).unwrap();
        match request.input {
            ResponseInput::Array(items) => {
                assert_eq!(items.len(), 2);
                assert_eq!(items[0].get("type").unwrap(), "message");
                assert_eq!(items[1].get("type").unwrap(), "function_call_output");
            }
            _ => panic!("Expected array input"),
        }
    }

    #[test]
    fn test_convert_input_to_messages() {
        let input = ResponseInput::String("Hello".to_string());
        let messages = input.into_messages();
        assert_eq!(messages.len(), 1);
        match &messages[0] {
            ChatMessage::User { content, .. } => assert_eq!(content, "Hello"),
            _ => panic!("Expected user message"),
        }
    }

    #[test]
    fn test_convert_array_with_function_call_output() {
        let input = ResponseInput::Array(vec![
            serde_json::json!({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}]
            }),
            serde_json::json!({
                "type": "function_call_output",
                "call_id": "call_123",
                "output": "Weather is sunny"
            }),
        ]);

        let messages = input.into_messages();
        assert_eq!(messages.len(), 2);

        match &messages[0] {
            ChatMessage::User { content, .. } => assert_eq!(content, "Hello"),
            _ => panic!("Expected user message"),
        }

        match &messages[1] {
            ChatMessage::Tool {
                content,
                tool_call_id,
            } => {
                assert_eq!(content, "Weather is sunny");
                assert_eq!(tool_call_id, "call_123");
            }
            _ => panic!("Expected tool message"),
        }
    }

    #[test]
    fn test_response_id_format() {
        let id = generate_response_id();
        assert!(id.starts_with("resp_"));
        assert_eq!(id.len(), 5 + 22);
    }

    #[test]
    fn test_message_id_format() {
        let id = generate_message_id();
        assert!(id.starts_with("msg_"));
        assert_eq!(id.len(), 4 + 22);
    }

    #[test]
    fn test_text_format_parsing() {
        let text_json = r#"{"format": {"type": "text"}}"#;
        let config: ResponseTextConfig = serde_json::from_str(text_json).unwrap();
        assert!(matches!(config.format, ResponseTextFormat::Text));

        let json_schema_json = r#"{"format": {"type": "json_schema", "name": "MySchema", "schema": {"type": "object"}}}"#;
        let config2: ResponseTextConfig = serde_json::from_str(json_schema_json).unwrap();
        assert!(matches!(
            config2.format,
            ResponseTextFormat::JsonSchema { .. }
        ));
    }

    #[test]
    fn test_tool_definition_parsing() {
        let json = r#"{
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {}}
        }"#;

        let tool: ResponseTool = serde_json::from_str(json).unwrap();
        assert_eq!(tool.tool_type, "function");
        assert_eq!(tool.name, "get_weather");
        assert_eq!(tool.description, Some("Get weather".to_string()));
    }

    #[test]
    fn test_reasoning_config_parsing() {
        let json = r#"{
            "effort": "medium",
            "summary": "auto"
        }"#;

        let reasoning: ReasoningConfig = serde_json::from_str(json).unwrap();
        assert_eq!(reasoning.effort, Some("medium".to_string()));
        assert_eq!(reasoning.summary, Some("auto".to_string()));
    }

    #[test]
    fn test_response_serialization() {
        let response = CreateResponseResponse {
            id: "resp_123".to_string(),
            object: "response".to_string(),
            status: "completed".to_string(),
            created_at: 1234567890,
            completed_at: Some(1234567891),
            model: "gpt-4".to_string(),
            output: vec![serde_json::json!({
                "type": "message",
                "id": "msg_123",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello", "annotations": []}]
            })],
            error: None,
            incomplete_details: None,
            instructions: None,
            max_output_tokens: None,
            previous_response_id: None,
            reasoning: None,
            store: Some(true),
            service_tier: None,
            temperature: Some(0.7),
            text: ResponseTextOutput {
                format: ResponseTextFormat::Text,
            },
            tool_choice: None,
            tools: None,
            top_p: None,
            truncation: None,
            usage: Some(ResponseUsage {
                input_tokens: 10,
                input_tokens_details: None,
                output_tokens: 20,
                output_tokens_details: None,
                total_tokens: 30,
            }),
            metadata: None,
            user: None,
            parallel_tool_calls: None,
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("resp_123"));
        assert!(json.contains("completed"));
        assert!(json.contains("\"store\":true"));
    }
}
