//! oobabooga (Text Generation WebUI) local LLM provider for LLMG
//!
//! Implements the Provider trait for oobabooga, a popular local LLM
//! inference WebUI that provides an API for local model execution.

use llmg_core::{
    provider::{LlmError, Provider},
    types::{ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse},
};

/// oobabooga API client
#[derive(Debug)]
pub struct OobaboogaClient {
    http_client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
}

/// oobabooga chat request format
#[derive(Debug, serde::Serialize)]
struct OobaboogaChatRequest {
    data: serde_json::Value,
}

/// oobabooga chat response format
#[derive(Debug, serde::Deserialize)]
struct OobaboogaChatResponse {
    data: Vec<serde_json::Value>,
}

impl OobaboogaClient {
    /// Create a new oobabooga client with default localhost URL
    pub fn new() -> Self {
        Self {
            http_client: reqwest::Client::new(),
            base_url: "http://localhost:7860/api/v1".to_string(),
            api_key: None,
        }
    }

    /// Create a new OobaboogaClient from environment variables.
    ///
    /// Reads `OOBABOOGA_BASE_URL` (default "http://localhost:7860/api/v1") and
    /// `OOBABOOGA_API_KEY` (optional).
    pub fn from_env() -> Self {
        let mut client = Self::new();
        if let Ok(base_url) = std::env::var("OOBABOOGA_BASE_URL") {
            client = client.with_base_url(base_url);
        }
        if let Ok(api_key) = std::env::var("OOBABOOGA_API_KEY") {
            client = client.with_api_key(api_key);
        }
        client
    }

    /// Create a new oobabooga client with API key
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Create with custom base URL
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Convert OpenAI format to oobabooga format
    fn convert_request(&self, request: ChatCompletionRequest) -> OobaboogaChatRequest {
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

        let data = serde_json::json!({
            "prompt": messages.iter()
                .filter_map(|m| m.get("content"))
                .filter_map(|c| c.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
            "model": request.model,
            "temperature": request.temperature.unwrap_or(0.7),
            "max_new_tokens": request.max_tokens.unwrap_or(256),
            "top_p": request.top_p,
            "stream": request.stream.unwrap_or(false),
        });

        OobaboogaChatRequest { data }
    }

    /// Convert oobabooga response to OpenAI format
    fn convert_response(
        &self,
        response: OobaboogaChatResponse,
        model: String,
    ) -> ChatCompletionResponse {
        let generated_text = response
            .data
            .first()
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        ChatCompletionResponse {
            id: format!("oobabooga-{})", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            choices: vec![llmg_core::types::Choice {
                index: 0,
                message: llmg_core::types::Message::Assistant {
                    content: if generated_text.is_empty() {
                        None
                    } else {
                        Some(generated_text)
                    },
                    refusal: None,
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: None,
        }
    }

    async fn make_request(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        let model = request.model.clone();
        let oobabooga_req = self.convert_request(request);
        let url = format!("{}/generate", self.base_url);

        let mut req_builder = self.http_client.post(&url).json(&oobabooga_req);

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

        let oobabooga_resp: OobaboogaChatResponse = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        Ok(self.convert_response(oobabooga_resp, model))
    }
}

impl Default for OobaboogaClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl Provider for OobaboogaClient {
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
        "oobabooga"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oobabooga_client_creation() {
        let client = OobaboogaClient::new();
        assert_eq!(client.provider_name(), "oobabooga");
        assert_eq!(client.base_url, "http://localhost:7860/api/v1");
    }

    #[test]
    fn test_oobabooga_custom_url() {
        let client = OobaboogaClient::new().with_base_url("http://custom-server:8080/api/v1");
        assert_eq!(client.base_url, "http://custom-server:8080/api/v1");
    }

    #[test]
    fn test_oobabooga_with_api_key() {
        let client = OobaboogaClient::new().with_api_key("test-key");
        assert_eq!(client.api_key, Some("test-key".to_string()));
    }

    #[test]
    fn test_request_conversion() {
        let client = OobaboogaClient::new();

        let request = ChatCompletionRequest {
            model: "mistral-7b-instruct".to_string(),
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
        };

        let oobabooga_req = client.convert_request(request);

        assert_eq!(oobabooga_req.data["model"], "mistral-7b-instruct");
    }
}
