use llmg_core::{
    provider::{ApiKeyCredentials, Credentials, LlmError, Provider},
    types::{ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse},
};

/// DeepInfra API client
#[derive(Debug)]
pub struct DeepInfraClient {
    http_client: reqwest::Client,
    base_url: String,
    credentials: Box<dyn Credentials>,
}

impl DeepInfraClient {
    /// Create a new DeepInfra client from environment
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("DEEPINFRA_API_KEY").map_err(|_| LlmError::AuthError)?;

        Ok(Self::new(api_key))
    }

    /// Create a new DeepInfra client with explicit API key
    pub fn new(api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();
        let credentials = Box::new(ApiKeyCredentials::bearer(api_key));

        Self {
            http_client: reqwest::Client::new(),
            base_url: "https://api.deepinfra.com/v1".to_string(),
            credentials,
        }
    }

    /// Create with custom base URL (for proxies or custom deployments)
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    async fn make_request(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        let url = format!("{}/chat/completions", self.base_url);

        let mut req = self
            .http_client
            .post(&url)
            .json(&request)
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

        response
            .json::<ChatCompletionResponse>()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))
    }
}

#[async_trait::async_trait]
impl Provider for DeepInfraClient {
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        self.make_request(request).await
    }

    async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        let url = format!("{}/embeddings", self.base_url);

        let mut req = self
            .http_client
            .post(&url)
            .json(&request)
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

        response
            .json::<EmbeddingResponse>()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))
    }
    fn provider_name(&self) -> &'static str {
        "deepinfra"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deepinfra_client_creation() {
        let client = DeepInfraClient::new("test-key");
        assert_eq!(client.provider_name(), "deepinfra");
    }

    #[test]
    fn test_from_env_missing_key() {
        // Temporarily remove env var
        let original = std::env::var("DEEPINFRA_API_KEY").ok();
        std::env::remove_var("DEEPINFRA_API_KEY");

        let result = DeepInfraClient::from_env();
        assert!(result.is_err());

        // Restore
        if let Some(key) = original {
            std::env::set_var("DEEPINFRA_API_KEY", key);
        }
    }
}

