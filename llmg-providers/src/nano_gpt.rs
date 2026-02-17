use llmg_core::{
    provider::{ApiKeyCredentials, Credentials, LlmError, Provider},
    types::{ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse},
};

/// Nano-GPT API client
#[derive(Debug)]
pub struct NanoGptClient {
    http_client: reqwest::Client,
    base_url: String,
    credentials: Box<dyn Credentials>,
}

impl NanoGptClient {
    /// Create a new Nano-GPT client from environment
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("NANO_GPT_API_KEY").map_err(|_| LlmError::AuthError)?;

        Ok(Self::new(api_key))
    }

    /// Create a new Nano-GPT client with explicit API key
    pub fn new(api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();
        let credentials = Box::new(ApiKeyCredentials::bearer(api_key));

        Self {
            http_client: reqwest::Client::new(),
            base_url: "https://api.nano-gpt.io/v1".to_string(),
            credentials,
        }
    }

    /// Create with custom base URL
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
impl Provider for NanoGptClient {
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
        "nano_gpt"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nano_gpt_client_creation() {
        let client = NanoGptClient::new("test-key");
        assert_eq!(client.provider_name(), "nano_gpt");
    }

    #[test]
    fn test_from_env_missing_key() {
        let original = std::env::var("NANO_GPT_API_KEY").ok();
        std::env::remove_var("NANO_GPT_API_KEY");

        let result = NanoGptClient::from_env();
        assert!(result.is_err());

        if let Some(key) = original {
            std::env::set_var("NANO_GPT_API_KEY", key);
        }
    }
}
