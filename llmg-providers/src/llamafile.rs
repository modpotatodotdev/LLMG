//! Llamafile local LLM provider for LLMG
//!
//! Implements the Provider trait for local Llamafile instances.
//! Llamafile provides OpenAI-compatible API for single-file LLM distribution.

use llmg_core::{
    provider::{LlmError, Provider},
    types::{ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse},
};

/// Llamafile API client
#[derive(Debug)]
pub struct LlamafileClient {
    http_client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
}

/// Llamafile chat request format (OpenAI-compatible)
#[derive(Debug, serde::Serialize)]
struct LlamafileChatRequest {
    model: String,
    messages: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
}

/// Llamafile chat response format (OpenAI-compatible)
#[derive(Debug, serde::Deserialize)]
struct LlamafileChatResponse {
    id: String,
    object: String,
    created: i64,
    model: String,
    choices: Vec<LlamafileChoice>,
    #[serde(default)]
    usage: Option<LlamafileUsage>,
}

#[derive(Debug, serde::Deserialize)]
struct LlamafileChoice {
    index: u32,
    message: LlamafileMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct LlamafileMessage {
    role: String,
    content: String,
}

#[derive(Debug, serde::Deserialize)]
struct LlamafileUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl LlamafileClient {
    /// Create a new Llamafile client with default localhost URL
    pub fn new() -> Self {
        Self {
            http_client: reqwest::Client::new(),
            base_url: "http://localhost:8080/v1".to_string(),
            api_key: None,
        }
    }

    /// Create a new LlamafileClient from environment variables.
    ///
    /// Reads `LLAMAFILE_BASE_URL` (default "http://localhost:8080/v1") and
    /// `LLAMAFILE_API_KEY` (optional).
    pub fn from_env() -> Self {
        let mut client = Self::new();
        if let Ok(base_url) = std::env::var("LLAMAFILE_BASE_URL") {
            client = client.with_base_url(base_url);
        }
        if let Ok(api_key) = std::env::var("LLAMAFILE_API_KEY") {
            client = client.with_api_key(api_key);
        }
        client
    }

    /// Create a new Llamafile client with API key
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Create with custom base URL
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Convert OpenAI format to Llamafile format
    fn convert_request(&self, request: ChatCompletionRequest) -> LlamafileChatRequest {
        let messages: Vec<serde_json::Value> = request
            .messages
            .into_iter()
            .filter_map(|msg| {
                let json_msg = match msg {
                    llmg_core::types::Message::System { content, .. } => {
                        serde_json::json!({ "role": "system", "content": content })
                    }
                    llmg_core::types::Message::User { content, .. } => {
                        serde_json::json!({ "role": "user", "content": content })
                    }
                    llmg_core::types::Message::Assistant { content, .. } => {
                        serde_json::json!({
                            "role": "assistant",
                            "content": content.unwrap_or_default()
                        })
                    }
                    _ => return None,
                };
                Some(json_msg)
            })
            .collect();

        LlamafileChatRequest {
            model: request.model,
            messages,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: request.stream,
            top_p: request.top_p,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            stop: request.stop,
        }
    }

    /// Convert Llamafile response to OpenAI format
    fn convert_response(&self, response: LlamafileChatResponse) -> ChatCompletionResponse {
        ChatCompletionResponse {
            id: response.id,
            object: response.object,
            created: response.created,
            model: response.model,
            choices: response
                .choices
                .into_iter()
                .map(|choice| llmg_core::types::Choice {
                    index: choice.index,
                    message: llmg_core::types::Message::Assistant {
                        content: Some(choice.message.content),
                        refusal: None,
                        tool_calls: None,
                    },
                    finish_reason: choice.finish_reason,
                })
                .collect(),
            usage: response.usage.map(|u| llmg_core::types::Usage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
            }),
        }
    }

    async fn make_request(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        let llamafile_req = self.convert_request(request);
        let url = format!("{}/chat/completions", self.base_url);

        let mut req_builder = self.http_client.post(&url).json(&llamafile_req);

        if let Some(ref key) = self.api_key {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", key));
        }

        let response = req_builder
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

        let llamafile_resp: LlamafileChatResponse = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        Ok(self.convert_response(llamafile_resp))
    }
}

impl Default for LlamafileClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl Provider for LlamafileClient {
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        self.make_request(request).await
    }

    async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        let url = format!("{}/embeddings", self.base_url);

        let mut req_builder = self.http_client.post(&url).json(&request);

        if let Some(ref key) = self.api_key {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", key));
        }

        let response = req_builder
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

        response
            .json::<EmbeddingResponse>()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))
    }
    fn provider_name(&self) -> &'static str {
        "llamafile"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llamafile_client_creation() {
        let client = LlamafileClient::new();
        assert_eq!(client.provider_name(), "llamafile");
        assert_eq!(client.base_url, "http://localhost:8080/v1");
    }

    #[test]
    fn test_llamafile_custom_url() {
        let client = LlamafileClient::new().with_base_url("http://custom-server:9000/v1");
        assert_eq!(client.base_url, "http://custom-server:9000/v1");
    }

    #[test]
    fn test_llamafile_with_api_key() {
        let client = LlamafileClient::new().with_api_key("test-key");
        assert_eq!(client.api_key, Some("test-key".to_string()));
    }

    #[test]
    fn test_request_conversion() {
        let client = LlamafileClient::new();

        let request = ChatCompletionRequest {
            model: "llama-3-70b".to_string(),
            messages: vec![llmg_core::types::Message::User {
                content: "Hello!".to_string(),
                name: None,
            }],
            temperature: Some(0.7),
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

        let llamafile_req = client.convert_request(request);

        assert_eq!(llamafile_req.model, "llama-3-70b");
        assert_eq!(llamafile_req.messages.len(), 1);
        assert_eq!(llamafile_req.temperature, Some(0.7));
        assert_eq!(llamafile_req.max_tokens, Some(100));
    }
}

