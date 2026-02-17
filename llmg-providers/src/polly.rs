use llmg_core::{
    provider::{ApiKeyCredentials, Credentials, LlmError, Provider},
    types::{ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse},
};

/// AWS Polly API client (Audio)
#[derive(Debug)]
pub struct PollyClient {
    http_client: reqwest::Client,
    base_url: String,
    credentials: Box<dyn Credentials>,
}

impl PollyClient {
    /// Create a new Polly client from environment
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("AWS_ACCESS_KEY_ID").map_err(|_| LlmError::AuthError)?;
        let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY").map_err(|_| LlmError::AuthError)?;
        let region = std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string());
        Ok(Self::new(api_key, secret_key, region))
    }

    pub fn new(
        api_key: impl Into<String>,
        _secret_key: impl Into<String>,
        region: impl Into<String>,
    ) -> Self {
        let region = region.into();
        Self {
            http_client: reqwest::Client::new(),
            base_url: format!("https://polly.{}.amazonaws.com", region),
            // For now, use basic auth header stub.
            // In a real implementation, this would use SigV4 like Bedrock.
            credentials: Box::new(ApiKeyCredentials::new(api_key)),
        }
    }
}

#[async_trait::async_trait]
impl Provider for PollyClient {
    async fn chat_completion(
        &self,
        _request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        Err(LlmError::UnsupportedFeature)
    }

    async fn embeddings(&self, _request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        Err(LlmError::UnsupportedFeature)
    }
    fn provider_name(&self) -> &'static str {
        "polly"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polly_client_creation() {
        let client = PollyClient::new("test-key", "test-secret", "us-east-1");
        assert_eq!(client.provider_name(), "polly");
    }

    #[test]
    fn test_from_env_missing_key() {
        let original_key = std::env::var("AWS_ACCESS_KEY_ID").ok();
        let original_secret = std::env::var("AWS_SECRET_ACCESS_KEY").ok();
        std::env::remove_var("AWS_ACCESS_KEY_ID");
        std::env::remove_var("AWS_SECRET_ACCESS_KEY");
        let result = PollyClient::from_env();
        assert!(result.is_err());
        if let Some(key) = original_key {
            std::env::set_var("AWS_ACCESS_KEY_ID", key);
        }
        if let Some(secret) = original_secret {
            std::env::set_var("AWS_SECRET_ACCESS_KEY", secret);
        }
    }
}
