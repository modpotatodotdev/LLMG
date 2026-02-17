use crate::streaming::ChatCompletionChunk;
use crate::types::{
    ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse,
};
use futures::Stream;
use std::fmt::Debug;
use std::future::Future;
use std::pin::Pin;

pub type ChatCompletionStream =
    Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk, LlmError>> + Send>>;

/// Core trait that all LLM providers must implement
#[async_trait::async_trait]
pub trait Provider: Send + Sync + Debug {
    /// Generate a chat completion
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError>;

    /// Stream a chat completion
    fn chat_completion_stream(
        &self,
        _request: ChatCompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ChatCompletionStream, LlmError>> + Send + '_>> {
        Box::pin(async { Err(LlmError::UnsupportedFeature) })
    }

    /// Generate embeddings for text
    async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError>;

    /// Get the list of models supported by this provider
    fn supported_models(&self) -> Vec<String> {
        vec![]
    }

    /// Dynamically list available models from the provider API
    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        Err(LlmError::UnsupportedFeature)
    }

    /// Get the provider name
    fn provider_name(&self) -> &'static str;
}

/// Error types for LLM operations
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("HTTP request failed: {0}")]
    HttpError(String),

    #[error("API error: {status} - {message}")]
    ApiError { status: u16, message: String },

    #[error("Authentication failed")]
    AuthError,

    #[error("Rate limit exceeded")]
    RateLimitError,

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Provider error: {0}")]
    ProviderError(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Unknown error: {0}")]
    Unknown(String),

    #[error("Feature not supported by this provider")]
    UnsupportedFeature,

    #[error("Resource not found")]
    NotFound,

    #[error("Internal provider error: {0}")]
    InternalError(String),

    #[error("Request timed out")]
    Timeout,
}

use std::sync::Arc;

/// Registry for managing multiple providers
pub struct ProviderRegistry {
    providers: Vec<Arc<dyn Provider>>,
}

impl ProviderRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
        }
    }

    /// Register a provider
    pub fn register(&mut self, provider: Arc<dyn Provider>) {
        self.providers.push(provider);
    }

    /// Get a provider by name
    pub fn get(&self, name: &str) -> Option<Arc<dyn Provider>> {
        self.providers
            .iter()
            .find(|p| p.provider_name() == name)
            .cloned()
    }

    /// List all registered providers
    pub fn list(&self) -> Vec<&'static str> {
        self.providers.iter().map(|p| p.provider_name()).collect()
    }

    /// Find provider that supports a specific model.
    pub fn find_by_model(&self, model: &str) -> Option<Arc<dyn Provider>> {
        self.providers
            .iter()
            .find(|p| p.supported_models().contains(&model.to_string()))
            .cloned()
    }
}

/// A provider that tries multiple other providers in sequence (fallbacks)
#[derive(Debug)]
pub struct FallbackProvider {
    providers: Vec<Box<dyn Provider>>,
}

impl FallbackProvider {
    pub fn new(providers: Vec<Box<dyn Provider>>) -> Self {
        Self { providers }
    }
}

#[async_trait::async_trait]
impl Provider for FallbackProvider {
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        let mut last_error = LlmError::ProviderError("No providers configured".to_string());

        for provider in &self.providers {
            match provider.chat_completion(request.clone()).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    tracing::warn!("Provider {} failed: {}", provider.provider_name(), e);
                    last_error = e;
                    // Only fallback on certain errors (e.g. RateLimit or ApiErrors)
                    // If it's a 400 Bad Request (InvalidRequest), we should probably stop.
                    if matches!(last_error, LlmError::InvalidRequest(_)) {
                        break;
                    }
                }
            }
        }

        Err(last_error)
    }

    async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        for provider in &self.providers {
            if let Ok(res) = provider.embeddings(request.clone()).await {
                return Ok(res);
            }
        }
        Err(LlmError::ProviderError(
            "All embedding providers failed".to_string(),
        ))
    }

    fn supported_models(&self) -> Vec<String> {
        self.providers
            .iter()
            .flat_map(|p| p.supported_models())
            .collect()
    }

    fn provider_name(&self) -> &'static str {
        "fallback"
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Credentials trait for authentication
pub trait Credentials: Send + Sync + Debug {
    /// Apply authentication to a request
    fn apply(&self, request: &mut reqwest::Request) -> Result<(), LlmError>;
}

/// Simple API key authentication
#[derive(Debug, Clone)]
pub struct ApiKeyCredentials {
    key: String,
    header_name: String,
}

impl ApiKeyCredentials {
    /// Create new API key credentials
    pub fn new(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            header_name: "Authorization".to_string(),
        }
    }

    /// Create with bearer token format
    pub fn bearer(key: impl Into<String>) -> Self {
        Self {
            key: format!("Bearer {}", key.into()),
            header_name: "Authorization".to_string(),
        }
    }

    /// Create with custom header
    pub fn with_header(key: impl Into<String>, header: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            header_name: header.into(),
        }
    }
}

impl Credentials for ApiKeyCredentials {
    fn apply(&self, request: &mut reqwest::Request) -> Result<(), LlmError> {
        request.headers_mut().insert(
            reqwest::header::HeaderName::from_bytes(self.header_name.as_bytes())
                .map_err(|e| LlmError::InvalidRequest(format!("Invalid header name: {}", e)))?,
            reqwest::header::HeaderValue::from_str(&self.key)
                .map_err(|e| LlmError::InvalidRequest(format!("Invalid header value: {}", e)))?,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct MockProvider;

    #[async_trait::async_trait]
    impl Provider for MockProvider {
        async fn chat_completion(
            &self,
            _request: ChatCompletionRequest,
        ) -> Result<ChatCompletionResponse, LlmError> {
            unimplemented!()
        }

        async fn embeddings(
            &self,
            _request: EmbeddingRequest,
        ) -> Result<EmbeddingResponse, LlmError> {
            unimplemented!()
        }

        fn supported_models(&self) -> Vec<String> {
            vec![]
        }

        fn provider_name(&self) -> &'static str {
            "mock"
        }
    }

    #[test]
    fn test_provider_registry() {
        let mut registry = ProviderRegistry::new();
        registry.register(Arc::new(MockProvider));

        assert_eq!(registry.list(), vec!["mock"]);
        assert!(registry.get("mock").is_some());
        assert!(registry.get("nonexistent").is_none());
    }
}
