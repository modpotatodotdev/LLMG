//! LiteLLM Proxy client for LLMG
//!
//! LiteLLM Proxy provides a unified gateway for accessing multiple LLM providers.
//! It uses an OpenAI-compatible API format.
//!
//! Environment variables:
//! - LITELLM_PROXY_API_KEY: Optional API key
//! - LITELLM_PROXY_URL: Optional custom base URL (default: http://localhost:4000/v1)

use llmg_core::{
    provider::{ApiKeyCredentials, Credentials, LlmError, Provider},
    types::{ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse},
};

/// LiteLLM Proxy API client
///
/// LiteLLM Proxy is a unified gateway for LLMs that provides:
/// - Access to 100+ models from various providers
/// - OpenAI-compatible API
/// - Configurable authentication
#[derive(Debug)]
pub struct LitellmProxyClient {
    http_client: reqwest::Client,
    base_url: String,
    credentials: Option<Box<dyn Credentials>>,
}

impl LitellmProxyClient {
    /// Create a new LiteLLM Proxy client from environment
    ///
    /// Optional: LITELLM_PROXY_API_KEY, LITELLM_PROXY_URL
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("LITELLM_PROXY_API_KEY").ok();
        let base_url = std::env::var("LITELLM_PROXY_URL")
            .unwrap_or_else(|_| "http://localhost:4000/v1".to_string());

        Ok(Self::new(api_key, base_url))
    }

    /// Create a new LiteLLM Proxy client with optional API key
    pub fn new(api_key: Option<String>, base_url: String) -> Self {
        let credentials =
            api_key.map(|key| Box::new(ApiKeyCredentials::bearer(key)) as Box<dyn Credentials>);

        Self {
            http_client: reqwest::Client::new(),
            base_url,
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

        let req_builder = self.http_client.post(&url).json(&request);

        let mut req = req_builder
            .build()
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if let Some(ref creds) = self.credentials {
            creds.apply(&mut req)?;
        }

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
impl Provider for LitellmProxyClient {
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        self.make_request(request).await
    }

    async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        let url = format!("{}/embeddings", self.base_url);

        let req_builder = self.http_client.post(&url).json(&request);

        let mut req = req_builder
            .build()
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if let Some(ref creds) = self.credentials {
            creds.apply(&mut req)?;
        }

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
        "litellm_proxy"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_litellm_proxy_client_creation() {
        let client = LitellmProxyClient::new(
            Some("test-key".to_string()),
            "http://localhost:4000/v1".to_string(),
        );
        assert_eq!(client.provider_name(), "litellm_proxy");
    }

    #[test]
    fn test_litellm_proxy_client_creation_no_key() {
        let client = LitellmProxyClient::new(None, "http://localhost:4000/v1".to_string());
        assert_eq!(client.provider_name(), "litellm_proxy");
    }

    #[test]
    fn test_default_base_url() {
        let client = LitellmProxyClient::new(None, "http://localhost:4000/v1".to_string());
        assert_eq!(client.base_url, "http://localhost:4000/v1");
    }

    #[test]
    fn test_with_base_url() {
        let client = LitellmProxyClient::new(None, "http://localhost:4000/v1".to_string())
            .with_base_url("http://custom:4000/v1");
        assert_eq!(client.base_url, "http://custom:4000/v1");
    }
}
