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
#[derive(Debug)]
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

/// Parse a model identifier in the format "provider/model"
/// Supports nested routing like "openrouter/openai/gpt-4"
///
/// Returns: (provider, model_name)
pub fn parse_model_id(model_id: &str) -> Result<(&str, String), String> {
    let parts: Vec<&str> = model_id.split('/').collect();

    if parts.len() < 2 {
        return Err("Model must be in format 'provider/model'".to_string());
    }

    let provider = parts[0];
    let model_name = parts[1..].join("/");

    if provider.is_empty() || model_name.is_empty() {
        return Err("Provider and model name cannot be empty".to_string());
    }

    Ok((provider, model_name))
}

/// A provider that acts as a router, delegating to other providers based on the model name.
/// Model names must be in the format `provider/model` (e.g., `openai/gpt-4`).
#[derive(Debug, Clone)]
pub struct RoutingProvider {
    registry: Arc<ProviderRegistry>,
}

impl RoutingProvider {
    /// Create a new routing provider with the given registry
    pub fn new(registry: ProviderRegistry) -> Self {
        Self {
            registry: Arc::new(registry),
        }
    }
}

#[async_trait::async_trait]
impl Provider for RoutingProvider {
    async fn chat_completion(
        &self,
        mut request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        // 1. Parse provider from model name
        let (provider_name, actual_model) =
            parse_model_id(&request.model).map_err(LlmError::InvalidRequest)?;

        // 2. Find provider
        let provider = self.registry.get(provider_name).ok_or_else(|| {
            LlmError::ProviderError(format!("Unknown provider: {}", provider_name))
        })?;

        // 3. Update request with actual model name (stripped of prefix)
        request.model = actual_model;

        // 4. Delegate
        provider.chat_completion(request).await
    }

    fn chat_completion_stream(
        &self,
        mut request: ChatCompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ChatCompletionStream, LlmError>> + Send + '_>> {
        // We need to capture the registry for the async block
        let registry = self.registry.clone();

        Box::pin(async move {
            let (provider_name, actual_model) =
                parse_model_id(&request.model).map_err(LlmError::InvalidRequest)?;

            let provider = registry.get(provider_name).ok_or_else(|| {
                LlmError::ProviderError(format!("Unknown provider: {}", provider_name))
            })?;

            request.model = actual_model;

            provider.chat_completion_stream(request).await
        })
    }

    async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        // Embeddings might not always have "model" in the same way, but usually do.
        // If the request format supports provider/model, we can route it.
        // However, EmbeddingRequest structure isn't shown fully here in context,
        // assuming it has a public `model` field like ChatCompletionRequest.

        let (provider_name, actual_model) =
            parse_model_id(&request.model).map_err(LlmError::InvalidRequest)?;

        let provider = self.registry.get(provider_name).ok_or_else(|| {
            LlmError::ProviderError(format!("Unknown provider: {}", provider_name))
        })?;

        // We need to clone the request to modify it, but EmbeddingRequest fields aren't fully visible.
        // Assuming we can create a new request or modify it.
        // The trait definition shows `request: EmbeddingRequest` (consumes it).
        // Let's rely on struct update syntax if fields are public.

        let mut new_request = request;
        new_request.model = actual_model;

        provider.embeddings(new_request).await
    }

    fn supported_models(&self) -> Vec<String> {
        // Return all models from all providers, prefixed with provider name
        let mut models = Vec::new();
        for provider in &self.registry.providers {
            let name = provider.provider_name();
            for model in provider.supported_models() {
                models.push(format!("{}/{}", name, model));
            }
        }
        models
    }

    fn provider_name(&self) -> &'static str {
        "router"
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
    fn test_parse_model_id_simple() {
        let result = parse_model_id("openai/gpt-4").unwrap();
        assert_eq!(result.0, "openai");
        assert_eq!(result.1, "gpt-4");
    }

    #[test]
    fn test_parse_model_id_nested() {
        let result = parse_model_id("openrouter/openai/gpt-4").unwrap();
        assert_eq!(result.0, "openrouter");
        assert_eq!(result.1, "openai/gpt-4");
    }

    #[test]
    fn test_parse_model_id_invalid() {
        assert!(parse_model_id("invalid").is_err());
        assert!(parse_model_id("/model").is_err());
        assert!(parse_model_id("provider/").is_err());
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
