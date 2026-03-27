//! WebSocket support for LLMG Gateway
//!
//! Implements the OpenAI Responses API WebSocket mode protocol for persistent
//! connections supporting multi-turn conversations with tool use.

use crate::routing::parse_model_id;
use crate::GatewayState;
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::Response,
};
use futures::StreamExt;
use llmg_core::types::{ChatCompletionRequest, Message as ChatMessage, Tool};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;

/// Generate a response ID matching OpenAI's format: resp_<22-char-base64>
fn generate_response_id() -> String {
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
    let uuid = uuid::Uuid::new_v4();
    let bytes = uuid.as_bytes();
    // Take first 16 bytes and encode to get ~22 chars
    format!("resp_{}", URL_SAFE_NO_PAD.encode(&bytes[..16]))
}

/// Generate a message ID matching OpenAI's format
fn generate_message_id() -> String {
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
    let uuid = uuid::Uuid::new_v4();
    let bytes = uuid.as_bytes();
    format!("msg_{}", URL_SAFE_NO_PAD.encode(&bytes[..16]))
}

/// WebSocket connection state for a single session
struct WsSession {
    previous_response_id: Option<String>,
    response_cache: std::collections::HashMap<String, ResponseCacheEntry>,
    conversation_history: Vec<ChatMessage>,
    #[allow(dead_code)]
    session_id: String,
    cancelled: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

/// Cached response data for conversation continuation
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct ResponseCacheEntry {
    /// The response ID
    id: String,
    /// The model used
    model: String,
    /// Messages from this response (for continuation)
    messages: Vec<ChatMessage>,
    /// Output text (for reference)
    output_text: String,
    /// Created timestamp
    created_at: u64,
}

/// Response format text configuration
#[derive(Debug, Deserialize, Default)]
struct TextFormatConfig {
    #[serde(default)]
    format: TextFormatType,
}

#[derive(Debug, Deserialize, Default)]
#[serde(tag = "type", rename_all = "snake_case")]
enum TextFormatType {
    #[default]
    Text,
    JsonObject,
    #[serde(rename = "json_schema")]
    JsonSchema {
        schema: serde_json::Value,
        name: String,
    },
}

/// Tool definition from Responses API format
#[derive(Debug, Deserialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
    #[serde(default)]
    #[allow(dead_code)]
    pub strict: Option<bool>,
}

/// Input can be a string or array of message objects
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum InputValue {
    String(String),
    Array(Vec<serde_json::Value>),
}

impl Default for InputValue {
    fn default() -> Self {
        InputValue::Array(Vec::new())
    }
}

/// Stream options for controlling streaming behavior
#[derive(Debug, Deserialize, Default)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

/// Tracks an in-progress function call during streaming
#[derive(Debug, Clone)]
struct StreamingToolCall {
    call_id: String,
    name: Option<String>,
    arguments: String,
    output_index: u32,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize, Default, Clone)]
pub struct ReasoningEffort {
    pub effort: Option<String>,
}

/// Client-sent event types
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ClientEvent {
    #[serde(rename = "response.create")]
    ResponseCreate {
        model: String,
        #[serde(default)]
        input: Box<InputValue>,
        previous_response_id: Option<String>,
        #[serde(default)]
        #[allow(dead_code)]
        conversation: Option<String>,
        #[serde(default)]
        store: Option<bool>,
        #[serde(default)]
        tools: Option<Vec<ToolDefinition>>,
        #[serde(default)]
        tool_choice: Option<Box<serde_json::Value>>,
        #[serde(default)]
        parallel_tool_calls: Option<bool>,
        #[serde(default)]
        instructions: Option<String>,
        #[serde(default)]
        temperature: Option<f32>,
        #[serde(default)]
        max_output_tokens: Option<u32>,
        #[serde(default)]
        top_p: Option<f32>,
        #[serde(default)]
        text: Option<Box<TextFormatConfig>>,
        #[serde(default)]
        #[allow(dead_code)]
        stream: Option<bool>,
        #[serde(default)]
        stream_options: Option<StreamOptions>,
        #[serde(default)]
        metadata: Option<Box<serde_json::Value>>,
        #[serde(default)]
        reasoning_effort: Option<String>,
        #[serde(default)]
        service_tier: Option<String>,
    },
    #[serde(rename = "response.compact")]
    ResponseCompact {
        #[allow(dead_code)]
        response_id: String,
    },
    #[serde(rename = "input_audio_buffer.append")]
    InputAudioBufferAppend {
        #[allow(dead_code)]
        audio: String,
    },
    #[serde(rename = "input_audio_buffer.commit")]
    InputAudioBufferCommit,
    #[serde(rename = "input_audio_buffer.clear")]
    InputAudioBufferClear,
    #[serde(rename = "conversation.item.create")]
    ConversationItemCreate { item: serde_json::Value },
    #[serde(rename = "conversation.item.truncate")]
    ConversationItemTruncate {
        item_id: String,
        #[allow(dead_code)]
        content_index: u32,
        truncate_offset: u32,
    },
    #[serde(rename = "conversation.item.delete")]
    ConversationItemDelete { item_id: String },
    #[serde(rename = "response.cancel")]
    ResponseCancel,
    #[serde(rename = "session.update")]
    SessionUpdate {
        #[allow(dead_code)]
        session: serde_json::Value,
    },
}

/// Server-sent event types
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ServerEvent {
    #[serde(rename = "response.created")]
    ResponseCreated { response: ResponseCreatedPayload },
    #[serde(rename = "response.done")]
    ResponseDone { response: Box<ResponseDonePayload> },
    #[serde(rename = "response.incomplete")]
    ResponseIncomplete { response_id: String, reason: String },
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
    #[serde(rename = "response.failed")]
    ResponseFailed {
        response_id: String,
        error: ErrorPayload,
    },
    #[serde(rename = "response.error")]
    ResponseError { error: ErrorPayload },
    #[serde(rename = "response.output_text.usage")]
    #[allow(dead_code)]
    ResponseOutputTextUsage {
        response_id: String,
        output_index: u32,
        content_index: u32,
        usage: UsagePayload,
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
    #[serde(rename = "session.created")]
    SessionCreated { session: SessionPayload },
    #[serde(rename = "session.updated")]
    SessionUpdated { session: SessionPayload },
    #[serde(rename = "input_audio_buffer.committed")]
    InputAudioBufferCommitted,
    #[serde(rename = "input_audio_buffer.cleared")]
    InputAudioBufferCleared,
    #[serde(rename = "conversation.item.created")]
    ConversationItemCreated { item: serde_json::Value },
    #[serde(rename = "conversation.item.truncated")]
    ConversationItemTruncated { item_id: String },
    #[serde(rename = "conversation.item.deleted")]
    ConversationItemDeleted { item_id: String },
    #[serde(rename = "rate_limits.updated")]
    #[allow(dead_code)]
    RateLimitsUpdated { rate_limits: Vec<RateLimitPayload> },
}

#[derive(Debug, Serialize)]
struct ResponseCreatedPayload {
    response: ResponseInfo,
}

#[derive(Debug, Serialize)]
struct ResponseInfo {
    id: String,
    object: String,
    status: String,
    created_at: u64,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_response_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct ResponseDonePayload {
    response: ResponseDetails,
}

#[derive(Debug, Serialize)]
struct ResponseDetails {
    id: String,
    object: String,
    status: String,
    created_at: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    completed_at: Option<u64>,
    model: String,
    output: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<ErrorPayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    incomplete_details: Option<IncompleteDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ReasoningPayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    text: TextFormatOutput,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    truncation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<UsagePayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
}

#[derive(Debug, Serialize)]
struct TextFormatOutput {
    format: TextFormatTypeOutput,
}

#[derive(Debug, Serialize, Default)]
#[serde(tag = "type", rename_all = "snake_case")]
enum TextFormatTypeOutput {
    #[default]
    Text,
    JsonObject,
    #[serde(rename = "json_schema")]
    JsonSchema {
        name: String,
        schema: serde_json::Value,
    },
}

#[derive(Debug, Serialize)]
struct IncompleteDetails {
    reason: String,
}

#[derive(Debug, Serialize)]
struct ReasoningPayload {
    effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    summary: Option<String>,
}

#[derive(Debug, Serialize)]
struct UsagePayload {
    input_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_tokens_details: Option<InputTokensDetails>,
    output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_tokens_details: Option<OutputTokensDetails>,
    total_tokens: u32,
}

#[derive(Debug, Serialize)]
struct InputTokensDetails {
    cached_tokens: u32,
}

#[derive(Debug, Serialize)]
struct OutputTokensDetails {
    reasoning_tokens: u32,
}

#[derive(Debug, Serialize)]
struct ErrorPayload {
    code: String,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    param: Option<String>,
}

#[derive(Debug, Serialize)]
struct SessionPayload {
    id: String,
    object: String,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
}

#[derive(Debug, Serialize)]
struct RateLimitPayload {
    limit: u32,
    remaining: u32,
    window: String,
}

/// Convert Responses API input (string or array) to chat messages
fn convert_input_to_messages(input: InputValue) -> Result<Vec<ChatMessage>, String> {
    match input {
        InputValue::String(s) => Ok(vec![ChatMessage::User {
            content: s,
            name: None,
        }]),
        InputValue::Array(items) => convert_input_array(items),
    }
}

/// Convert input items array from Responses API format to ChatCompletionRequest
fn convert_input_array(input: Vec<serde_json::Value>) -> Result<Vec<ChatMessage>, String> {
    let mut messages = Vec::new();

    for item in input {
        let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");

        match item_type {
            "message" => {
                let role = item.get("role").and_then(|v| v.as_str()).unwrap_or("user");

                let content = extract_content_from_message(&item);

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
                        let tool_calls = extract_tool_calls_from_message(&item);
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

    Ok(messages)
}

/// Extract tool calls from a message's content array for conversation continuation
fn extract_tool_calls_from_message(
    item: &serde_json::Value,
) -> Option<Vec<llmg_core::types::ToolCall>> {
    let content_array = item.get("content").and_then(|v| v.as_array())?;
    let mut tool_calls = Vec::new();

    for c in content_array {
        if c.get("type").and_then(|v| v.as_str()) == Some("function_call") {
            let call_id = c.get("id")?.as_str()?.to_string();
            let name = c.get("name")?.as_str()?.to_string();
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

/// Extract text content from a message content array
/// Handles both input_text and input_image blocks
/// Returns content with images converted to data URLs for providers that support vision
fn extract_content_from_message(item: &serde_json::Value) -> String {
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
                    if let Some(image_url) = c
                        .get("image_url")
                        .and_then(|v| v.get("url"))
                        .and_then(|v| v.as_str())
                    {
                        parts.push(image_url.to_string());
                    } else if let Some(image_data) = c.get("image_data").and_then(|v| v.as_str()) {
                        parts.push(image_data.to_string());
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

fn convert_conversation_item(item: &serde_json::Value) -> Result<Option<ChatMessage>, String> {
    let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");

    match item_type {
        "message" => {
            let role = item.get("role").and_then(|v| v.as_str()).unwrap_or("user");

            let content = extract_content_from_message(item);

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
                    let tool_calls = extract_tool_calls_from_message(item);
                    ChatMessage::Assistant {
                        content: Some(content),
                        refusal: None,
                        tool_calls,
                    }
                }
                _ => return Ok(None),
            };
            Ok(Some(msg))
        }
        "function_call_output" => {
            let output = item.get("output").and_then(|v| v.as_str()).unwrap_or("");
            let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");

            Ok(Some(ChatMessage::Tool {
                content: output.to_string(),
                tool_call_id: call_id.to_string(),
            }))
        }
        "compaction" => Ok(None),
        _ => Ok(None),
    }
}

/// Handle incoming WebSocket messages
async fn handle_ws_message(
    socket: &mut WebSocket,
    state: Arc<GatewayState>,
    session: &mut WsSession,
    msg: Message,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    match msg {
        Message::Text(text) => {
            let event: ClientEvent = match serde_json::from_str(&text) {
                Ok(e) => e,
                Err(e) => {
                    let error_event = ServerEvent::ResponseError {
                        error: ErrorPayload {
                            code: "parse_error".to_string(),
                            message: format!("Failed to parse event: {}", e),
                            param: None,
                        },
                    };
                    socket
                        .send(Message::Text(serde_json::to_string(&error_event)?))
                        .await?;
                    return Ok(());
                }
            };

            match event {
                ClientEvent::ResponseCreate {
                    model,
                    input,
                    previous_response_id,
                    conversation: _,
                    store,
                    tools,
                    tool_choice,
                    parallel_tool_calls,
                    instructions,
                    temperature,
                    max_output_tokens,
                    top_p,
                    text,
                    stream: _,
                    stream_options,
                    metadata,
                    reasoning_effort,
                    service_tier,
                } => {
                    // Handle previous_response_id - prepend cached conversation
                    let mut conversation_messages = Vec::new();

                    // Inject previous_response_id content if provided
                    if let Some(prev_id) = &previous_response_id {
                        if let Some(cached) = session.response_cache.get(prev_id) {
                            conversation_messages.extend(cached.messages.clone());
                        }
                    }

                    let response_id = generate_response_id();
                    let created_at = chrono::Utc::now().timestamp() as u64;

                    let created_event = ServerEvent::ResponseCreated {
                        response: ResponseCreatedPayload {
                            response: ResponseInfo {
                                id: response_id.clone(),
                                object: "response".to_string(),
                                status: "in_progress".to_string(),
                                created_at,
                                model: model.clone(),
                                previous_response_id: previous_response_id.clone(),
                            },
                        },
                    };
                    socket
                        .send(Message::Text(serde_json::to_string(&created_event)?))
                        .await?;

                    // Convert input to messages
                    let input_messages = match convert_input_to_messages(*input) {
                        Ok(m) => m,
                        Err(e) => {
                            let error_event = ServerEvent::ResponseError {
                                error: ErrorPayload {
                                    code: "invalid_request".to_string(),
                                    message: e,
                                    param: Some("input".to_string()),
                                },
                            };
                            socket
                                .send(Message::Text(serde_json::to_string(&error_event)?))
                                .await?;
                            return Ok(());
                        }
                    };

                    // Prepend instructions as system message if provided
                    let mut all_messages = conversation_messages;
                    let instructions_content = instructions.clone().unwrap_or_default();

                    if !instructions_content.is_empty() {
                        all_messages.insert(
                            0,
                            ChatMessage::System {
                                content: instructions_content,
                                name: None,
                            },
                        );
                    }
                    all_messages.extend(input_messages);

                    let resolved_model = state
                        .config
                        .aliases
                        .get(&model)
                        .cloned()
                        .unwrap_or_else(|| model.clone());

                    // Convert tools from Responses API format to Chat Completions format
                    let converted_tools = tools.map(|tools_list| {
                        let mut saw_unsupported = false;
                        let result: Vec<_> = tools_list
                            .into_iter()
                            .filter_map(|t| {
                                match t.tool_type.as_str() {
                                    "function" => Some(Tool {
                                        r#type: "function".to_string(),
                                        function: llmg_core::types::FunctionDefinition {
                                            name: t.name,
                                            description: t.description,
                                            parameters: t.parameters.unwrap_or(serde_json::json!({})),
                                        },
                                    }),
                                    "computer_use_preview" | "web_search_preview" | "file_search" => {
                                        saw_unsupported = true;
                                        None
                                    }
                                    _ => {
                                        saw_unsupported = true;
                                        None
                                    }
                                }
                            })
                            .collect();
                        if saw_unsupported {
                            tracing::warn!("Some tool types (computer_use_preview, web_search_preview, file_search) are not supported by the gateway and were ignored");
                        }
                        result
                    });

                    // Parse tool_choice from JSON value
                    let parsed_tool_choice = tool_choice.and_then(|tc| {
                        let tc = *tc;
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
                    });

                    let request = ChatCompletionRequest {
                        model: resolved_model,
                        messages: all_messages.clone(),
                        stream: Some(true),
                        temperature,
                        max_tokens: max_output_tokens,
                        top_p,
                        frequency_penalty: None,
                        presence_penalty: None,
                        stop: None,
                        user: None,
                        tools: converted_tools,
                        tool_choice: parsed_tool_choice,
                    };

                    // Build text format output
                    let text_format = text.map(|t| t.format).unwrap_or_default();

                    // Store response info for potential conversation continuation
                    session.response_cache.insert(
                        response_id.clone(),
                        ResponseCacheEntry {
                            id: response_id.clone(),
                            model: model.clone(),
                            messages: all_messages,
                            output_text: String::new(),
                            created_at,
                        },
                    );
                    session.previous_response_id = previous_response_id.clone();

                    handle_streaming_response(
                        socket,
                        state,
                        request,
                        response_id,
                        created_at,
                        model.clone(),
                        store,
                        parallel_tool_calls,
                        text_format,
                        stream_options,
                        metadata.map(|m| *m),
                        instructions,
                        reasoning_effort,
                        session.previous_response_id.clone(),
                        service_tier,
                        session.cancelled.clone(),
                    )
                    .await?;
                }
                ClientEvent::ResponseCancel => {
                    session
                        .cancelled
                        .store(true, std::sync::atomic::Ordering::SeqCst);
                    let incomplete_event = ServerEvent::ResponseIncomplete {
                        response_id: session.previous_response_id.clone().unwrap_or_default(),
                        reason: "cancelled".to_string(),
                    };
                    socket
                        .send(Message::Text(serde_json::to_string(&incomplete_event)?))
                        .await?;
                }
                ClientEvent::ResponseCompact { response_id: _ } => {
                    let error_event = ServerEvent::ResponseError {
                        error: ErrorPayload {
                            code: "not_implemented".to_string(),
                            message: "Response compaction requires LLM summarization".to_string(),
                            param: None,
                        },
                    };
                    socket
                        .send(Message::Text(serde_json::to_string(&error_event)?))
                        .await?;
                }
                ClientEvent::ConversationItemCreate { item } => {
                    match convert_conversation_item(&item) {
                        Ok(Some(msg)) => {
                            session.conversation_history.push(msg);
                        }
                        Ok(None) => {}
                        Err(e) => {
                            let error_event = ServerEvent::ResponseError {
                                error: ErrorPayload {
                                    code: "invalid_item".to_string(),
                                    message: e,
                                    param: Some("item".to_string()),
                                },
                            };
                            socket
                                .send(Message::Text(serde_json::to_string(&error_event)?))
                                .await?;
                            return Ok(());
                        }
                    }
                    let item_id = item.get("id").and_then(|v| v.as_str()).unwrap_or("unknown");
                    let created_event = ServerEvent::ConversationItemCreated {
                        item: json!({
                            "id": item_id,
                            "type": "message",
                            "status": "completed"
                        }),
                    };
                    socket
                        .send(Message::Text(serde_json::to_string(&created_event)?))
                        .await?;
                }
                ClientEvent::ConversationItemTruncate {
                    item_id,
                    content_index: _,
                    truncate_offset,
                } => {
                    let original_len = session.conversation_history.len();
                    for msg in session.conversation_history.iter_mut() {
                        match msg {
                            ChatMessage::Assistant {
                                content: Some(c), ..
                            } => {
                                if c.len() > truncate_offset as usize {
                                    c.truncate(truncate_offset as usize);
                                }
                            }
                            ChatMessage::User { content, .. } => {
                                if content.len() > truncate_offset as usize {
                                    content.truncate(truncate_offset as usize);
                                }
                            }
                            _ => {}
                        }
                    }
                    let truncated_event = ServerEvent::ConversationItemTruncated {
                        item_id: item_id.clone(),
                    };
                    socket
                        .send(Message::Text(serde_json::to_string(&truncated_event)?))
                        .await?;
                    tracing::debug!("Truncated item {} in {} messages", item_id, original_len);
                }
                ClientEvent::ConversationItemDelete { item_id } => {
                    let original_len = session.conversation_history.len();
                    session.conversation_history.retain(|m| match m {
                        ChatMessage::Assistant { content, .. } => !content
                            .as_ref()
                            .map(|c| c.contains(&item_id))
                            .unwrap_or(false),
                        ChatMessage::User { content, .. } => !content.contains(&item_id),
                        _ => true,
                    });
                    let deleted_event = ServerEvent::ConversationItemDeleted {
                        item_id: item_id.clone(),
                    };
                    socket
                        .send(Message::Text(serde_json::to_string(&deleted_event)?))
                        .await?;
                    tracing::debug!(
                        "Deleted item {}, removed {} messages",
                        item_id,
                        original_len - session.conversation_history.len()
                    );
                }
                ClientEvent::SessionUpdate { session: _ } => {
                    let update_event = ServerEvent::SessionUpdated {
                        session: SessionPayload {
                            id: format!("sess_{}", uuid::Uuid::new_v4()),
                            object: "session".to_string(),
                            model: "unknown".to_string(),
                            instructions: None,
                        },
                    };
                    socket
                        .send(Message::Text(serde_json::to_string(&update_event)?))
                        .await?;
                }
                ClientEvent::InputAudioBufferAppend { audio: _ } => {
                    let error_event = ServerEvent::ResponseError {
                        error: ErrorPayload {
                            code: "not_implemented".to_string(),
                            message: "Audio input not yet supported".to_string(),
                            param: None,
                        },
                    };
                    socket
                        .send(Message::Text(serde_json::to_string(&error_event)?))
                        .await?;
                }
                ClientEvent::InputAudioBufferCommit => {
                    let committed_event = ServerEvent::InputAudioBufferCommitted;
                    socket
                        .send(Message::Text(serde_json::to_string(&committed_event)?))
                        .await?;
                }
                ClientEvent::InputAudioBufferClear => {
                    let cleared_event = ServerEvent::InputAudioBufferCleared;
                    socket
                        .send(Message::Text(serde_json::to_string(&cleared_event)?))
                        .await?;
                }
            }
        }
        Message::Close(_) => {
            return Err("Connection closed".into());
        }
        _ => {}
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn handle_streaming_response(
    socket: &mut WebSocket,
    state: Arc<GatewayState>,
    mut request: ChatCompletionRequest,
    response_id: String,
    created_at: u64,
    model: String,
    store: Option<bool>,
    parallel_tool_calls: Option<bool>,
    text_format: TextFormatType,
    stream_options: Option<StreamOptions>,
    metadata: Option<serde_json::Value>,
    instructions: Option<String>,
    reasoning_effort: Option<String>,
    previous_response_id: Option<String>,
    service_tier: Option<String>,
    cancelled: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let request_max_tokens = request.max_tokens;
    let request_temperature = request.temperature;
    let request_top_p = request.top_p;
    let request_tools = request.tools.clone();
    let request_tool_choice = request.tool_choice.clone();
    let request_reasoning_effort = reasoning_effort.clone();
    let include_usage = stream_options
        .as_ref()
        .map(|o| o.include_usage)
        .unwrap_or(false);

    let (provider_name, model_name) = match parse_model_id(&request.model) {
        Ok((p, m)) => (p.to_string(), m),
        Err(e) => {
            let error_event = ServerEvent::ResponseFailed {
                response_id: response_id.clone(),
                error: ErrorPayload {
                    code: "invalid_model".to_string(),
                    message: format!("{:?}", e),
                    param: Some("model".to_string()),
                },
            };
            socket
                .send(Message::Text(serde_json::to_string(&error_event)?))
                .await?;
            return Ok(());
        }
    };

    request.model = model_name;

    let provider = match state.registry.get(&provider_name) {
        Some(p) => p.clone(),
        None => {
            let error_event = ServerEvent::ResponseFailed {
                response_id: response_id.clone(),
                error: ErrorPayload {
                    code: "unknown_provider".to_string(),
                    message: format!("Unknown provider: {}", provider_name),
                    param: None,
                },
            };
            socket
                .send(Message::Text(serde_json::to_string(&error_event)?))
                .await?;
            return Ok(());
        }
    };

    let stream = match provider.chat_completion_stream(request).await {
        Ok(s) => s,
        Err(e) => {
            let error_event = ServerEvent::ResponseFailed {
                response_id: response_id.clone(),
                error: ErrorPayload {
                    code: "provider_error".to_string(),
                    message: e.to_string(),
                    param: None,
                },
            };
            socket
                .send(Message::Text(serde_json::to_string(&error_event)?))
                .await?;
            return Ok(());
        }
    };

    let message_id = generate_message_id();
    let mut full_text = String::new();
    let mut item_added_sent = false;
    let mut tool_calls: Vec<StreamingToolCall> = Vec::new();
    let mut output_items: Vec<serde_json::Value> = Vec::new();
    let mut usage_input_tokens = 0u32;
    let mut usage_output_tokens = 0u32;
    let mut seen_first_chunk = false;

    use futures::StreamExt;
    let mut stream = stream;

    while let Some(chunk_result) = stream.next().await {
        if cancelled.load(std::sync::atomic::Ordering::SeqCst) {
            break;
        }

        match chunk_result {
            Ok(chunk) => {
                if include_usage {
                    if let Some(usage) = &chunk.usage {
                        usage_input_tokens = usage.prompt_tokens;
                        usage_output_tokens = usage.completion_tokens;
                    }
                }

                for choice in &chunk.choices {
                    if !item_added_sent {
                        item_added_sent = true;
                    }

                    if let Some(tc_deltas) = &choice.delta.tool_calls {
                        for tc_delta in tc_deltas {
                            let call_id = tc_delta.id.clone().unwrap_or_else(|| {
                                if tool_calls.is_empty() {
                                    generate_message_id()
                                } else {
                                    tool_calls.last().unwrap().call_id.clone()
                                }
                            });

                            let is_new_call = !tool_calls.iter().any(|t| t.call_id == call_id);

                            if is_new_call {
                                let output_index = output_items.len() as u32;
                                let func_name =
                                    tc_delta.function.as_ref().and_then(|f| f.name.clone());

                                let item_added = ServerEvent::ResponseOutputItemAdded {
                                    response_id: response_id.clone(),
                                    output_index,
                                    item: json!({
                                        "type": "function_call",
                                        "id": call_id.clone(),
                                        "name": func_name.clone().unwrap_or_default(),
                                        "arguments": "",
                                        "status": "in_progress"
                                    }),
                                };
                                socket
                                    .send(Message::Text(serde_json::to_string(&item_added)?))
                                    .await?;

                                tool_calls.push(StreamingToolCall {
                                    call_id: call_id.clone(),
                                    name: func_name,
                                    arguments: String::new(),
                                    output_index,
                                });
                            }

                            if let Some(tc) = tool_calls.iter_mut().find(|t| t.call_id == call_id) {
                                if let Some(name) =
                                    tc_delta.function.as_ref().and_then(|f| f.name.clone())
                                {
                                    tc.name = Some(name);
                                }
                                if let Some(args) =
                                    tc_delta.function.as_ref().and_then(|f| f.arguments.clone())
                                {
                                    tc.arguments.push_str(&args);
                                    let delta_event =
                                        ServerEvent::ResponseFunctionCallArgumentsDelta {
                                            response_id: response_id.clone(),
                                            output_index: tc.output_index,
                                            call_id: call_id.clone(),
                                            delta: args,
                                        };
                                    socket
                                        .send(Message::Text(serde_json::to_string(&delta_event)?))
                                        .await?;
                                }
                            }
                        }
                    }

                    if let Some(content) = &choice.delta.content {
                        if !seen_first_chunk && !tool_calls.is_empty() {
                            seen_first_chunk = true;
                        }
                        full_text.push_str(content);
                        let content_index = 0u32;
                        let delta_event = ServerEvent::ResponseOutputTextDelta {
                            response_id: response_id.clone(),
                            output_index: 0,
                            content_index,
                            delta: content.clone(),
                        };
                        socket
                            .send(Message::Text(serde_json::to_string(&delta_event)?))
                            .await?;
                    }
                }
            }
            Err(e) => {
                let error_event = ServerEvent::ResponseFailed {
                    response_id: response_id.clone(),
                    error: ErrorPayload {
                        code: "stream_error".to_string(),
                        message: e.to_string(),
                        param: None,
                    },
                };
                socket
                    .send(Message::Text(serde_json::to_string(&error_event)?))
                    .await?;
                return Ok(());
            }
        }
    }

    let completed_at = chrono::Utc::now().timestamp() as u64;

    for tc in &tool_calls {
        let done_event = ServerEvent::ResponseFunctionCallArgumentsDone {
            response_id: response_id.clone(),
            output_index: tc.output_index,
            call_id: tc.call_id.clone(),
            arguments: tc.arguments.clone(),
        };
        socket
            .send(Message::Text(serde_json::to_string(&done_event)?))
            .await?;

        let func_name = tc.name.clone().unwrap_or_default();
        let item_done = ServerEvent::ResponseOutputItemDone {
            response_id: response_id.clone(),
            output_index: tc.output_index,
            item: json!({
                "type": "function_call",
                "id": tc.call_id,
                "name": func_name,
                "arguments": tc.arguments,
                "status": "completed"
            }),
        };
        socket
            .send(Message::Text(serde_json::to_string(&item_done)?))
            .await?;

        output_items.push(json!({
            "type": "function_call",
            "id": tc.call_id,
            "name": func_name,
            "arguments": tc.arguments
        }));
    }

    if !full_text.is_empty() || tool_calls.is_empty() {
        let text_content_index = 0u32;

        if !full_text.is_empty() {
            let done_event = ServerEvent::ResponseOutputTextDone {
                response_id: response_id.clone(),
                output_index: text_content_index,
                content_index: 0,
                text: full_text.clone(),
            };
            socket
                .send(Message::Text(serde_json::to_string(&done_event)?))
                .await?;
        }

        let msg_output_index = 0u32;

        let item_done = ServerEvent::ResponseOutputItemDone {
            response_id: response_id.clone(),
            output_index: msg_output_index,
            item: json!({
                "type": "message",
                "id": message_id.clone(),
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": full_text,
                    "annotations": []
                }]
            }),
        };
        socket
            .send(Message::Text(serde_json::to_string(&item_done)?))
            .await?;

        output_items.insert(
            0,
            json!({
                "type": "message",
                "id": message_id,
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": full_text,
                    "annotations": []
                }]
            }),
        );
    }

    let usage = if include_usage && (usage_input_tokens > 0 || usage_output_tokens > 0) {
        Some(UsagePayload {
            input_tokens: usage_input_tokens,
            input_tokens_details: None,
            output_tokens: usage_output_tokens,
            output_tokens_details: None,
            total_tokens: usage_input_tokens.saturating_add(usage_output_tokens),
        })
    } else {
        None
    };

    let text_format_output = TextFormatOutput {
        format: match text_format {
            TextFormatType::Text => TextFormatTypeOutput::Text,
            TextFormatType::JsonObject => TextFormatTypeOutput::JsonObject,
            TextFormatType::JsonSchema { name, schema } => {
                TextFormatTypeOutput::JsonSchema { name, schema }
            }
        },
    };

    let done_payload = ServerEvent::ResponseDone {
        response: Box::new(ResponseDonePayload {
            response: ResponseDetails {
                id: response_id.clone(),
                object: "response".to_string(),
                status: "completed".to_string(),
                created_at,
                completed_at: Some(completed_at),
                model: model.clone(),
                output: output_items,
                error: None,
                incomplete_details: None,
                instructions: instructions.clone(),
                max_output_tokens: request_max_tokens,
                parallel_tool_calls,
                previous_response_id: previous_response_id.clone(),
                reasoning: request_reasoning_effort.as_ref().map(|e| ReasoningPayload {
                    effort: Some(e.clone()),
                    summary: None,
                }),
                store,
                service_tier,
                temperature: request_temperature,
                text: text_format_output,
                tool_choice: request_tool_choice.as_ref().map(|tc| json!(tc)),
                tools: request_tools.as_ref().map(|t| {
                    t.iter()
                        .map(|tool| {
                            json!({
                                "type": "function",
                                "function": {
                                    "name": tool.function.name,
                                    "description": tool.function.description,
                                    "parameters": tool.function.parameters
                                }
                            })
                        })
                        .collect()
                }),
                top_p: request_top_p,
                truncation: None,
                usage,
                metadata,
                user: None,
            },
        }),
    };
    socket
        .send(Message::Text(serde_json::to_string(&done_payload)?))
        .await?;

    Ok(())
}

/// POST handler for Responses API that upgrades to WebSocket
pub async fn responses_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<GatewayState>>,
) -> Response {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: Arc<GatewayState>) {
    let mut socket = socket;
    let session_id = format!("sess_{}", uuid::Uuid::new_v4());
    let cancelled = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut session = WsSession {
        previous_response_id: None,
        response_cache: std::collections::HashMap::new(),
        conversation_history: Vec::new(),
        session_id: session_id.clone(),
        cancelled: cancelled.clone(),
    };

    let session_event = ServerEvent::SessionCreated {
        session: SessionPayload {
            id: session_id,
            object: "session".to_string(),
            model: "unknown".to_string(),
            instructions: None,
        },
    };

    if let Ok(event_str) = serde_json::to_string(&session_event) {
        let _ = socket.send(Message::Text(event_str)).await;
    }

    while let Some(msg) = socket.next().await {
        match msg {
            Ok(msg) => {
                if let Err(e) =
                    handle_ws_message(&mut socket, state.clone(), &mut session, msg).await
                {
                    tracing::error!("WebSocket error: {}", e);
                    break;
                }
            }
            Err(e) => {
                tracing::error!("WebSocket receive error: {}", e);
                break;
            }
        }
    }

    tracing::info!("WebSocket connection closed");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_client_event() {
        let json = r#"{"type":"response.create","model":"openai/gpt-4","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Hello"}]}],"previous_response_id":null}"#;
        let event: ClientEvent = serde_json::from_str(json).unwrap();

        match event {
            ClientEvent::ResponseCreate {
                model,
                input,
                previous_response_id,
                ..
            } => {
                assert_eq!(model, "openai/gpt-4");
                assert!(previous_response_id.is_none());
                if let InputValue::Array(arr) = input {
                    assert_eq!(arr.len(), 1);
                } else {
                    panic!("Expected array input");
                }
            }
            _ => panic!("Expected ResponseCreate"),
        }
    }

    #[test]
    fn test_convert_input_string() {
        let input = InputValue::String("Hello".to_string());
        let messages = convert_input_to_messages(input).unwrap();
        assert_eq!(messages.len(), 1);

        match &messages[0] {
            ChatMessage::User { content, .. } => {
                assert_eq!(content, "Hello");
            }
            _ => panic!("Expected User message"),
        }
    }

    #[test]
    fn test_convert_input_messages() {
        let input = InputValue::Array(vec![json!({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello"}]
        })]);

        let messages = convert_input_to_messages(input).unwrap();
        assert_eq!(messages.len(), 1);

        match &messages[0] {
            ChatMessage::User { content, .. } => {
                assert_eq!(content, "Hello");
            }
            _ => panic!("Expected User message"),
        }
    }

    #[test]
    fn test_response_id_format() {
        let id = generate_response_id();
        assert!(id.starts_with("resp_"));
        assert_eq!(id.len(), 5 + 22); // "resp_" + 22 char base64
    }

    #[test]
    fn test_message_id_format() {
        let id = generate_message_id();
        assert!(id.starts_with("msg_"));
        assert_eq!(id.len(), 4 + 22); // "msg_" + 22 char base64
    }

    #[test]
    fn test_parse_response_create_with_all_fields() {
        let json = r#"{
            "type": "response.create",
            "model": "gpt-4o",
            "input": "Hello",
            "max_output_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "parallel_tool_calls": true,
            "stream_options": {"include_usage": true}
        }"#;

        let event: ClientEvent = serde_json::from_str(json).unwrap();

        match event {
            ClientEvent::ResponseCreate {
                model,
                max_output_tokens,
                temperature,
                top_p,
                parallel_tool_calls,
                stream_options,
                ..
            } => {
                assert_eq!(model, "gpt-4o");
                assert_eq!(max_output_tokens, Some(100));
                assert_eq!(temperature, Some(0.7));
                assert_eq!(top_p, Some(0.9));
                assert_eq!(parallel_tool_calls, Some(true));
                assert_eq!(stream_options.unwrap().include_usage, true);
            }
            _ => panic!("Expected ResponseCreate"),
        }
    }

    #[test]
    fn test_parse_reasoning_effort() {
        let json = r#"{
            "type": "response.create",
            "model": "o1-preview",
            "input": "Hello",
            "reasoning_effort": "high"
        }"#;

        let event: ClientEvent = serde_json::from_str(json).unwrap();

        match event {
            ClientEvent::ResponseCreate {
                model,
                reasoning_effort,
                ..
            } => {
                assert_eq!(model, "o1-preview");
                assert!(reasoning_effort.is_some());
                assert_eq!(reasoning_effort.unwrap(), "high");
            }
            _ => panic!("Expected ResponseCreate"),
        }
    }

    #[test]
    fn test_parse_service_tier() {
        let json = r#"{
            "type": "response.create",
            "model": "gpt-4o",
            "input": "Hello",
            "service_tier": "auto"
        }"#;

        let event: ClientEvent = serde_json::from_str(json).unwrap();

        match event {
            ClientEvent::ResponseCreate {
                model,
                service_tier,
                ..
            } => {
                assert_eq!(model, "gpt-4o");
                assert_eq!(service_tier, Some("auto".to_string()));
            }
            _ => panic!("Expected ResponseCreate"),
        }
    }

    #[test]
    fn test_parse_conversation_item_create() {
        let json = r#"{
            "type": "conversation.item.create",
            "item": {
                "id": "msg_123",
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}]
            }
        }"#;

        let event: ClientEvent = serde_json::from_str(json).unwrap();

        match event {
            ClientEvent::ConversationItemCreate { item } => {
                assert_eq!(item.get("id").and_then(|v| v.as_str()), Some("msg_123"));
                assert_eq!(item.get("type").and_then(|v| v.as_str()), Some("message"));
            }
            _ => panic!("Expected ConversationItemCreate"),
        }
    }

    #[test]
    fn test_parse_conversation_item_truncate() {
        let json = r#"{
            "type": "conversation.item.truncate",
            "item_id": "msg_123",
            "content_index": 0,
            "truncate_offset": 50
        }"#;

        let event: ClientEvent = serde_json::from_str(json).unwrap();

        match event {
            ClientEvent::ConversationItemTruncate {
                item_id,
                content_index,
                truncate_offset,
            } => {
                assert_eq!(item_id, "msg_123");
                assert_eq!(content_index, 0);
                assert_eq!(truncate_offset, 50);
            }
            _ => panic!("Expected ConversationItemTruncate"),
        }
    }

    #[test]
    fn test_parse_conversation_item_delete() {
        let json = r#"{
            "type": "conversation.item.delete",
            "item_id": "msg_123"
        }"#;

        let event: ClientEvent = serde_json::from_str(json).unwrap();

        match event {
            ClientEvent::ConversationItemDelete { item_id } => {
                assert_eq!(item_id, "msg_123");
            }
            _ => panic!("Expected ConversationItemDelete"),
        }
    }

    #[test]
    fn test_parse_input_audio_buffer_events() {
        let append_json = r#"{"type":"input_audio_buffer.append","audio":"base64data"}"#;
        let append_event: ClientEvent = serde_json::from_str(append_json).unwrap();
        assert!(matches!(
            append_event,
            ClientEvent::InputAudioBufferAppend { .. }
        ));

        let commit_json = r#"{"type":"input_audio_buffer.commit"}"#;
        let commit_event: ClientEvent = serde_json::from_str(commit_json).unwrap();
        assert!(matches!(commit_event, ClientEvent::InputAudioBufferCommit));

        let clear_json = r#"{"type":"input_audio_buffer.clear"}"#;
        let clear_event: ClientEvent = serde_json::from_str(clear_json).unwrap();
        assert!(matches!(clear_event, ClientEvent::InputAudioBufferClear));
    }

    #[test]
    fn test_convert_conversation_item_message() {
        let item = json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "input_text", "text": "Hello"}]
        });

        let result = convert_conversation_item(&item);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[test]
    fn test_convert_conversation_item_function_output() {
        let item = json!({
            "type": "function_call_output",
            "output": "result data",
            "call_id": "call_123"
        });

        let result = convert_conversation_item(&item);
        assert!(result.is_ok());
        let msg = result.unwrap().unwrap();
        assert!(matches!(msg, ChatMessage::Tool { .. }));
    }

    #[test]
    fn test_text_format_config_parsing() {
        let text_json = json!({
            "format": {"type": "text"}
        });
        let config: TextFormatConfig = serde_json::from_value(text_json).unwrap();
        assert!(matches!(config.format, TextFormatType::Text));

        let json_schema_json = json!({
            "format": {
                "type": "json_schema",
                "schema": {"type": "object"},
                "name": "MySchema"
            }
        });
        let config2: TextFormatConfig = serde_json::from_value(json_schema_json).unwrap();
        assert!(matches!(config2.format, TextFormatType::JsonSchema { .. }));
    }

    #[test]
    fn test_tool_definition_parsing() {
        let json = json!({
            "type": "function",
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {"type": "object", "properties": {}}
        });

        let tool: ToolDefinition = serde_json::from_value(json).unwrap();
        assert_eq!(tool.tool_type, "function");
        assert_eq!(tool.name, "get_weather");
        assert_eq!(
            tool.description,
            Some("Get weather for a location".to_string())
        );
    }
}
