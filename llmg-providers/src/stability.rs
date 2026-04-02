use llmg_core::{
    provider::{ApiKeyCredentials, Credentials, LlmError, Provider},
    types::{ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse},
};

/// Stability AI API client (Vision)
#[derive(Debug)]
pub struct StabilityAiClient {
    http_client: reqwest::Client,
    base_url: String,
    credentials: Box<dyn Credentials>,
}

impl StabilityAiClient {
    /// Create a new Stability AI client from environment
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("STABILITY_API_KEY").map_err(|_| LlmError::AuthError)?;
        Ok(Self::new(api_key))
    }

    pub fn new(api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();
        Self {
            http_client: reqwest::Client::new(),
            base_url: "https://api.stability.ai/v1".to_string(),
            credentials: Box::new(ApiKeyCredentials::bearer(api_key)),
        }
    }
}

#[async_trait::async_trait]
impl Provider for StabilityAiClient {
    async fn chat_completion(
        &self,
        _request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        // Stability is Image generation.
        Err(LlmError::UnsupportedFeature)
    }

    async fn embeddings(&self, _request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        Err(LlmError::UnsupportedFeature)
    }
    fn provider_name(&self) -> &'static str {
        "stability"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stability_client_creation() {
        let client = StabilityAiClient::new("test-key");
        assert_eq!(client.provider_name(), "stability");
    }

    #[test]
    fn test_from_env_missing_key() {
        let original = std::env::var("STABILITY_API_KEY").ok();
        std::env::remove_var("STABILITY_API_KEY");
        let result = StabilityAiClient::from_env();
        assert!(result.is_err());
        if let Some(key) = original {
            std::env::set_var("STABILITY_API_KEY", key);
        }
    }
}

