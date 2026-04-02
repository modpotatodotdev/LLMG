//! Petals decentralized LLM provider for LLMG
//!
//! Implements the Provider trait for Petals, a decentralized peer-to-peer
//! LLM network that allows running large models collectively across consumer devices.

use llmg_core::{
    provider::{LlmError, Provider},
    types::{ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse},
};

/// Petals API client
#[derive(Debug)]
pub struct PetalsClient {
    http_client: reqwest::Client,
    base_url: String,
}

/// Petals chat request format
#[derive(Debug, serde::Serialize)]
struct PetalsChatRequest {
    model: String,
    inputs: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<PetalsParameters>,
}

#[derive(Debug, serde::Serialize)]
struct PetalsParameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_new_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
}

/// Petals chat response format
#[derive(Debug, serde::Deserialize)]
struct PetalsChatResponse {
    generated_text: String,
}

impl PetalsClient {
    /// Create a new Petals client with default API URL
    pub fn new() -> Self {
        Self {
            http_client: reqwest::Client::new(),
            base_url: "https://petals.ml/api/v1".to_string(),
        }
    }

    /// Create a new PetalsClient from environment variables.
    ///
    /// Reads `PETALS_BASE_URL` (default "https://petals.ml/api/v1").
    pub fn from_env() -> Self {
        let mut client = Self::new();
        if let Ok(base_url) = std::env::var("PETALS_BASE_URL") {
            client = client.with_base_url(base_url);
        }
        client
    }

    /// Create with custom base URL (for self-hosted Petals cluster)
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Convert OpenAI format to Petals format
    fn convert_request(&self, request: ChatCompletionRequest) -> PetalsChatRequest {
        let input_text = request
            .messages
            .iter()
            .map(|msg| match msg {
                llmg_core::types::Message::System { content, .. } => {
                    format!("System: {}\n", content)
                }
                llmg_core::types::Message::User { content, .. } => format!("User: {}\n", content),
                llmg_core::types::Message::Assistant { content, .. } => {
                    format!("Assistant: {}\n", content.as_deref().unwrap_or(""))
                }
                _ => String::new(),
            })
            .collect::<Vec<_>>()
            .join("");

        let parameters = if request.temperature.is_some()
            || request.max_tokens.is_some()
            || request.top_p.is_some()
        {
            Some(PetalsParameters {
                temperature: request.temperature,
                max_new_tokens: request.max_tokens,
                top_p: request.top_p,
            })
        } else {
            None
        };

        PetalsChatRequest {
            model: request.model,
            inputs: input_text,
            parameters,
        }
    }

    /// Convert Petals response to OpenAI format
    fn convert_response(
        &self,
        response: PetalsChatResponse,
        model: String,
    ) -> ChatCompletionResponse {
        ChatCompletionResponse {
            id: format!("petals-{})", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            choices: vec![llmg_core::types::Choice {
                index: 0,
                message: llmg_core::types::Message::Assistant {
                    content: Some(response.generated_text),
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
        let petals_req = self.convert_request(request);
        let url = format!("{}/generate", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .json(&petals_req)
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

        let petals_resp: PetalsChatResponse = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        Ok(self.convert_response(petals_resp, model))
    }
}

impl Default for PetalsClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl Provider for PetalsClient {
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        self.make_request(request).await
    }

    async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        let url = format!("{}/embeddings", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .json(&request)
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
        "petals"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_petals_client_creation() {
        let client = PetalsClient::new();
        assert_eq!(client.provider_name(), "petals");
        assert_eq!(client.base_url, "https://petals.ml/api/v1");
    }

    #[test]
    fn test_petals_custom_url() {
        let client = PetalsClient::new().with_base_url("http://custom-cluster:8080/api/v1");
        assert_eq!(client.base_url, "http://custom-cluster:8080/api/v1");
    }

    #[test]
    fn test_request_conversion() {
        let client = PetalsClient::new();

        let request = ChatCompletionRequest {
            model: "petals-team/StableBeluga2".to_string(),
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

        let petals_req = client.convert_request(request);

        assert_eq!(petals_req.model, "petals-team/StableBeluga2");
        assert!(petals_req.inputs.contains("User: Hello!"));
        assert!(petals_req.parameters.is_some());
    }
}

