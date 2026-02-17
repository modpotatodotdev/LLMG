use llmg_core::{
    provider::{ApiKeyCredentials, Credentials, LlmError, Provider},
    types::{ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse},
};

/// Firecrawl API client (Search/Scrape)
#[derive(Debug)]
pub struct FirecrawlClient {
    http_client: reqwest::Client,
    base_url: String,
    credentials: Box<dyn Credentials>,
}

impl FirecrawlClient {
    /// Create a new Firecrawl client from environment
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("FIRECRAWL_API_KEY").map_err(|_| LlmError::AuthError)?;
        Ok(Self::new(api_key))
    }

    pub fn new(api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();
        Self {
            http_client: reqwest::Client::new(),
            base_url: "https://api.firecrawl.dev/v1".to_string(),
            credentials: Box::new(ApiKeyCredentials::bearer(api_key)),
        }
    }
}

#[async_trait::async_trait]
impl Provider for FirecrawlClient {
    async fn chat_completion(
        &self,
        _request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        // Firecrawl is search.
        Err(LlmError::UnsupportedFeature)
    }

    async fn embeddings(&self, _request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        Err(LlmError::UnsupportedFeature)
    }
    fn provider_name(&self) -> &'static str {
        "firecrawl"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_firecrawl_client_creation() {
        let client = FirecrawlClient::new("test-key");
        assert_eq!(client.provider_name(), "firecrawl");
    }

    #[test]
    fn test_from_env_missing_key() {
        let original = std::env::var("FIRECRAWL_API_KEY").ok();
        std::env::remove_var("FIRECRAWL_API_KEY");
        let result = FirecrawlClient::from_env();
        assert!(result.is_err());
        if let Some(key) = original {
            std::env::set_var("FIRECRAWL_API_KEY", key);
        }
    }
}
