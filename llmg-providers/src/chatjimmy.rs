//! ChatJimmy API client for LLMG
//!
//! Reverse-engineered client for the chatjimmy.ai chat API.
//! This is a free, unauthenticated provider that streams responses
//! via SSE (text/event-stream) with a trailing `<|stats|>…<|/stats|>` JSON block.
//!
//! **No API key required** — this is a public endpoint.
//!
//! Default model: `llama3.1-8B`

use futures::StreamExt;
use llmg_core::{
    provider::{ChatCompletionStream, LlmError, Provider},
    streaming::{ChatCompletionChunk, DeltaContent},
    types::{
        ChatCompletionRequest, ChatCompletionResponse, Choice, EmbeddingRequest, EmbeddingResponse,
        Message, Usage,
    },
};
use regex::Regex;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const API_URL: &str = "https://chatjimmy.ai/api/chat";
const DEFAULT_MODEL: &str = "llama3.1-8B";
const DEFAULT_TOP_K: u32 = 8;

// ---------------------------------------------------------------------------
// Internal request / response types
// ---------------------------------------------------------------------------

/// A message in ChatJimmy's wire format.
#[derive(Debug, Clone, serde::Serialize)]
struct JimmyMessage {
    role: String,
    content: String,
}

/// Chat options sent alongside each request.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct JimmyChatOptions {
    selected_model: String,
    system_prompt: String,
    top_k: u32,
}

/// The full request body for the ChatJimmy API.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct JimmyRequest {
    messages: Vec<JimmyMessage>,
    chat_options: JimmyChatOptions,
    attachment: Option<serde_json::Value>,
}

/// Performance statistics embedded in the `<|stats|>…<|/stats|>` block.
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[allow(dead_code)]
struct JimmyStats {
    #[serde(default)]
    created_at: f64,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    done_reason: String,
    #[serde(default)]
    total_duration: f64,
    #[serde(default)]
    ttft: f64,
    #[serde(default)]
    prefill_tokens: u32,
    #[serde(default)]
    prefill_rate: f64,
    #[serde(default)]
    decode_tokens: u32,
    #[serde(default)]
    decode_rate: f64,
    #[serde(default)]
    total_tokens: u32,
    #[serde(default)]
    total_time: f64,
    #[serde(default)]
    roundtrip_time: u32,
    #[serde(default)]
    status: u32,
    #[serde(default)]
    reason: String,
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// ChatJimmy API client.
///
/// A free, unauthenticated LLM provider powered by `chatjimmy.ai`.
/// Streams responses via raw SSE text and parses the trailing stats block.
///
/// # Default model
///
/// `llama3.1-8B`
#[derive(Debug)]
pub struct ChatJimmyClient {
    http_client: reqwest::Client,
    api_url: String,
    default_model: String,
    system_prompt: String,
    top_k: u32,
}

impl ChatJimmyClient {
    /// Create a new ChatJimmy client with default settings.
    ///
    /// No API key is required — this endpoint is public.
    pub fn new() -> Self {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("Accept", "*/*".parse().unwrap());
        headers.insert("Accept-Language", "en-US,en;q=0.9".parse().unwrap());
        headers.insert("Content-Type", "application/json".parse().unwrap());
        headers.insert("Origin", "https://chatjimmy.ai".parse().unwrap());
        headers.insert("Referer", "https://chatjimmy.ai/".parse().unwrap());
        headers.insert("DNT", "1".parse().unwrap());
        headers.insert("Sec-GPC", "1".parse().unwrap());
        headers.insert("Sec-Fetch-Dest", "empty".parse().unwrap());
        headers.insert("Sec-Fetch-Mode", "cors".parse().unwrap());
        headers.insert("Sec-Fetch-Site", "same-origin".parse().unwrap());
        headers.insert("Cache-Control", "no-cache".parse().unwrap());
        headers.insert("Pragma", "no-cache".parse().unwrap());

        let http_client = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .expect("failed to build reqwest client");

        Self {
            http_client,
            api_url: API_URL.to_string(),
            default_model: DEFAULT_MODEL.to_string(),
            system_prompt: String::new(),
            top_k: DEFAULT_TOP_K,
        }
    }

    /// Create from environment (no-op since no API key needed, but follows convention).
    pub fn from_env() -> Result<Self, LlmError> {
        Ok(Self::new())
    }

    /// Set a custom API URL (e.g. for a local proxy).
    pub fn with_api_url(mut self, url: impl Into<String>) -> Self {
        self.api_url = url.into();
        self
    }

    /// Set the default model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    /// Set a system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Set the top-K sampling parameter.
    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = top_k;
        self
    }

    /// Convert LLMG messages into ChatJimmy wire-format messages.
    fn convert_messages(messages: &[Message]) -> Vec<JimmyMessage> {
        messages
            .iter()
            .map(|m| match m {
                Message::System { content, .. } => JimmyMessage {
                    role: "system".to_string(),
                    content: content.clone(),
                },
                Message::User { content, .. } => JimmyMessage {
                    role: "user".to_string(),
                    content: content.clone(),
                },
                Message::Assistant { content, .. } => JimmyMessage {
                    role: "assistant".to_string(),
                    content: content.clone().unwrap_or_default(),
                },
                Message::Tool { content, .. } => JimmyMessage {
                    role: "user".to_string(),
                    content: content.clone(),
                },
            })
            .collect()
    }

    /// Parse the raw streamed text, stripping the `<|stats|>…<|/stats|>` block.
    fn parse_response(full_text: &str) -> (String, Option<JimmyStats>) {
        let re = Regex::new(r"<\|stats\|>(.*?)<\|/stats\|>").unwrap();
        if let Some(caps) = re.captures(full_text) {
            let stats_start = caps.get(0).unwrap().start();
            let content = &full_text[..stats_start];
            let stats = serde_json::from_str::<JimmyStats>(caps.get(1).unwrap().as_str()).ok();
            (content.to_string(), stats)
        } else {
            (full_text.to_string(), None)
        }
    }

    /// Build the request payload for ChatJimmy's API.
    fn build_payload(&self, request: &ChatCompletionRequest) -> JimmyRequest {
        let model = if request.model.is_empty() {
            self.default_model.clone()
        } else {
            request.model.clone()
        };

        JimmyRequest {
            messages: Self::convert_messages(&request.messages),
            chat_options: JimmyChatOptions {
                selected_model: model,
                system_prompt: self.system_prompt.clone(),
                top_k: self.top_k,
            },
            attachment: None,
        }
    }

    /// Send a request and collect the full streamed response into a string.
    async fn make_request(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        let model = if request.model.is_empty() {
            self.default_model.clone()
        } else {
            request.model.clone()
        };

        let payload = self.build_payload(&request);

        let response = self
            .http_client
            .post(&self.api_url)
            .json(&payload)
            .send()
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

        // Stream the full text from the SSE response
        let full_text = response
            .text()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        let (content, stats) = Self::parse_response(&full_text);

        // Map stats into Usage if available
        let usage = stats.as_ref().map(|s| Usage {
            prompt_tokens: s.prefill_tokens,
            completion_tokens: s.decode_tokens,
            total_tokens: s.total_tokens,
        });

        let id = ChatCompletionChunk::generate_id();

        Ok(ChatCompletionResponse {
            id,
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            choices: vec![Choice {
                index: 0,
                message: Message::Assistant {
                    content: Some(content),
                    refusal: None,
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage,
        })
    }
}

impl Default for ChatJimmyClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl Provider for ChatJimmyClient {
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        self.make_request(request).await
    }

    fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<ChatCompletionStream, LlmError>> + Send + '_>,
    > {
        Box::pin(async move {
            let model = if request.model.is_empty() {
                self.default_model.clone()
            } else {
                request.model.clone()
            };

            let payload = self.build_payload(&request);
            let chunk_id = ChatCompletionChunk::generate_id();
            let model_clone = model.clone();

            let response = self
                .http_client
                .post(&self.api_url)
                .json(&payload)
                .send()
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

            let byte_stream = response.bytes_stream();
            let sent_role = false;
            let chunk_id_clone = chunk_id.clone();
            let model_for_stream = model_clone.clone();

            let stream = byte_stream.map(move |result| {
                match result {
                    Ok(bytes) => {
                        let text = String::from_utf8_lossy(&bytes).to_string();

                        // Filter out the stats block
                        let clean = if text.contains("<|stats|>") {
                            text.split("<|stats|>").next().unwrap_or("").to_string()
                        } else {
                            text
                        };

                        if clean.is_empty() {
                            // Final chunk
                            Ok(ChatCompletionChunk::final_chunk(
                                chunk_id_clone.clone(),
                                model_for_stream.clone(),
                                "stop",
                            ))
                        } else {
                            let delta = if !sent_role {
                                // We can't mutate `sent_role` in an FnMut inside
                                // futures::StreamExt::map easily, so we just always
                                // send content deltas. The role delta is optional.
                                DeltaContent {
                                    role: Some("assistant".to_string()),
                                    content: Some(clean),
                                    tool_calls: None,
                                }
                            } else {
                                DeltaContent::content(clean)
                            };

                            Ok(ChatCompletionChunk::new(
                                chunk_id_clone.clone(),
                                model_for_stream.clone(),
                                0,
                                delta,
                                None,
                            ))
                        }
                    }
                    Err(e) => Err(LlmError::HttpError(e.to_string())),
                }
            });

            // Mark role as sent after first chunk (the stream handles it inline)
            let _ = sent_role;

            Ok(Box::pin(stream) as ChatCompletionStream)
        })
    }

    async fn embeddings(&self, _request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        Err(LlmError::ProviderError(
            "ChatJimmy does not support embeddings".to_string(),
        ))
    }

    fn supported_models(&self) -> Vec<String> {
        vec!["llama3.1-8B".to_string()]
    }

    fn provider_name(&self) -> &'static str {
        "chatjimmy"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chatjimmy_client_creation() {
        let client = ChatJimmyClient::new();
        assert_eq!(client.provider_name(), "chatjimmy");
        assert_eq!(client.default_model, "llama3.1-8B");
    }

    #[test]
    fn test_from_env() {
        // Should always succeed since no API key is needed
        let result = ChatJimmyClient::from_env();
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_methods() {
        let client = ChatJimmyClient::new()
            .with_model("custom-model")
            .with_system_prompt("You are helpful.")
            .with_top_k(16)
            .with_api_url("http://localhost:8080/api/chat");

        assert_eq!(client.default_model, "custom-model");
        assert_eq!(client.system_prompt, "You are helpful.");
        assert_eq!(client.top_k, 16);
        assert_eq!(client.api_url, "http://localhost:8080/api/chat");
    }

    #[test]
    fn test_parse_response_with_stats() {
        let raw = r#"Hello, world!<|stats|>{"done":true,"total_tokens":42,"decode_tokens":10,"prefill_tokens":32}<|/stats|>"#;
        let (content, stats) = ChatJimmyClient::parse_response(raw);
        assert_eq!(content, "Hello, world!");
        let stats = stats.unwrap();
        assert!(stats.done);
        assert_eq!(stats.total_tokens, 42);
        assert_eq!(stats.decode_tokens, 10);
        assert_eq!(stats.prefill_tokens, 32);
    }

    #[test]
    fn test_parse_response_without_stats() {
        let raw = "Just some text without stats.";
        let (content, stats) = ChatJimmyClient::parse_response(raw);
        assert_eq!(content, "Just some text without stats.");
        assert!(stats.is_none());
    }

    #[test]
    fn test_convert_messages() {
        let messages = vec![
            Message::System {
                content: "You are a helpful assistant.".to_string(),
                name: None,
            },
            Message::User {
                content: "Hello!".to_string(),
                name: None,
            },
        ];

        let jimmy_msgs = ChatJimmyClient::convert_messages(&messages);
        assert_eq!(jimmy_msgs.len(), 2);
        assert_eq!(jimmy_msgs[0].role, "system");
        assert_eq!(jimmy_msgs[0].content, "You are a helpful assistant.");
        assert_eq!(jimmy_msgs[1].role, "user");
        assert_eq!(jimmy_msgs[1].content, "Hello!");
    }

    #[test]
    fn test_supported_models() {
        let client = ChatJimmyClient::new();
        let models = client.supported_models();
        assert!(models.contains(&"llama3.1-8B".to_string()));
    }
}
