//! OpenRouter API client for LLMG
//!
//! OpenRouter provides a unified interface for accessing many LLM providers.
//! It uses an OpenAI-compatible API format.
//!
//! Environment variables:
//! - OPENROUTER_API_KEY: Required API key
//! - OPENROUTER_API_BASE: Optional custom base URL (default: https://openrouter.ai/api/v1)
//! - OPENROUTER_APP_NAME: Optional app name for rankings
//! - OPENROUTER_HTTP_REFERER: Optional HTTP referer

use llmg_core::{
    provider::{ApiKeyCredentials, Credentials, LlmError, Provider},
    types::{ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse},
};
use std::sync::Arc;
// use serde::Serialize; // removed unused import

/// OpenRouter API client
///
/// OpenRouter is a unified interface for LLMs that provides:
/// - Access to 100+ models from various providers
/// - OpenAI-compatible API
/// - Automatic fallback and routing
#[derive(Debug, Clone)]
pub struct OpenRouterClient {
    http_client: reqwest::Client,
    base_url: String,
    credentials: Arc<dyn Credentials>,
    app_name: Option<String>,
    http_referer: Option<String>,
}

/// OpenRouter-specific request extensions
#[derive(Debug, Clone, Default)]
pub struct OpenRouterExtras {
    /// Provider selection preferences (e.g., ["Anthropic", "OpenAI"])
    pub provider: Option<serde_json::Value>,
    /// Transformations to apply
    pub transforms: Option<Vec<String>>,
    /// Route configuration
    pub route: Option<String>,
    /// Models to include/exclude
    pub models: Option<Vec<String>>,
}

/// OpenRouter chat request with extensions
#[derive(Debug, serde::Serialize)]
struct OpenRouterChatRequest {
    #[serde(flatten)]
    base: ChatCompletionRequest,
    #[serde(skip_serializing_if = "Option::is_none")]
    provider: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    transforms: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    route: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    models: Option<Vec<String>>,
}

impl OpenRouterClient {
    /// Create a new OpenRouter client from environment
    ///
    /// Required: OPENROUTER_API_KEY
    /// Optional: OPENROUTER_API_BASE, OPENROUTER_APP_NAME, OPENROUTER_HTTP_REFERER
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("OPENROUTER_API_KEY").map_err(|_| LlmError::AuthError)?;

        let base_url = std::env::var("OPENROUTER_API_BASE")
            .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());

        let app_name = std::env::var("OPENROUTER_APP_NAME").ok();
        let http_referer = std::env::var("OPENROUTER_HTTP_REFERER").ok();

        Ok(Self::with_config(api_key, base_url, app_name, http_referer))
    }

    /// Create a new OpenRouter client with explicit API key
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_config(
            api_key,
            "https://openrouter.ai/api/v1".to_string(),
            None,
            None,
        )
    }

    /// Create with custom configuration
    pub fn with_config(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        app_name: Option<String>,
        http_referer: Option<String>,
    ) -> Self {
        let api_key = api_key.into();

        Self {
            http_client: reqwest::Client::new(),
            base_url: base_url.into(),
            credentials: Arc::new(ApiKeyCredentials::bearer(api_key)),
            app_name,
            http_referer,
        }
    }

    /// Create with custom base URL
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set app name for OpenRouter rankings
    pub fn with_app_name(mut self, name: impl Into<String>) -> Self {
        self.app_name = Some(name.into());
        self
    }

    /// Set HTTP referer for OpenRouter rankings
    pub fn with_http_referer(mut self, referer: impl Into<String>) -> Self {
        self.http_referer = Some(referer.into());
        self
    }

    /// Build request with OpenRouter-specific headers
    fn build_request(
        &self,
        request: ChatCompletionRequest,
        extras: Option<OpenRouterExtras>,
    ) -> Result<reqwest::Request, LlmError> {
        let url = format!("{}/chat/completions", self.base_url);

        // Convert to OpenRouter format with extensions
        let openrouter_req = if let Some(extras) = extras {
            OpenRouterChatRequest {
                base: request,
                provider: extras.provider,
                transforms: extras.transforms,
                route: extras.route,
                models: extras.models,
            }
        } else {
            OpenRouterChatRequest {
                base: request,
                provider: None,
                transforms: None,
                route: None,
                models: None,
            }
        };

        let mut req_builder = self.http_client.post(&url).json(&openrouter_req);

        // Add OpenRouter-specific headers
        if let Some(ref app_name) = self.app_name {
            req_builder = req_builder.header("X-Title", app_name);
        }

        if let Some(ref referer) = self.http_referer {
            req_builder = req_builder.header("HTTP-Referer", referer);
        }

        let mut req = req_builder
            .build()
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        self.credentials.apply(&mut req)?;

        Ok(req)
    }

    async fn make_request(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        let req = self.build_request(request, None)?;

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

    /// Make a chat completion with OpenRouter-specific extras
    pub async fn chat_completion_with_extras(
        &self,
        request: ChatCompletionRequest,
        extras: OpenRouterExtras,
    ) -> Result<ChatCompletionResponse, LlmError> {
        let req = self.build_request(request, Some(extras))?;

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
impl Provider for OpenRouterClient {
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
        "openrouter"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmg_core::types::Message;

    #[test]
    fn test_openrouter_client_creation() {
        let client = OpenRouterClient::new("test-key");
        assert_eq!(client.provider_name(), "openrouter");
    }

    #[test]
    fn test_from_env_missing_key() {
        // Temporarily remove env var
        let original = std::env::var("OPENROUTER_API_KEY").ok();
        std::env::remove_var("OPENROUTER_API_KEY");

        let result = OpenRouterClient::from_env();
        assert!(result.is_err());

        // Restore
        if let Some(key) = original {
            std::env::set_var("OPENROUTER_API_KEY", key);
        }
    }

    #[test]
    fn test_custom_config() {
        let client = OpenRouterClient::with_config(
            "test-key",
            "https://custom.openrouter.ai/api/v1",
            Some("MyApp".to_string()),
            Some("https://myapp.com".to_string()),
        );

        assert_eq!(client.base_url, "https://custom.openrouter.ai/api/v1");
        assert_eq!(client.app_name, Some("MyApp".to_string()));
        assert_eq!(client.http_referer, Some("https://myapp.com".to_string()));
    }

    #[test]
    fn test_extras_builder() {
        let extras = OpenRouterExtras {
            provider: Some(serde_json::json!({"order": ["Anthropic", "OpenAI"]})),
            transforms: Some(vec!["middle-out".to_string()]),
            route: Some("fallback".to_string()),
            models: Some(vec!["anthropic/claude-3-opus".to_string()]),
        };

        let request = ChatCompletionRequest {
            model: "anthropic/claude-3-opus".to_string(),
            messages: vec![Message::User {
                content: "Hello".to_string(),
                name: None,
            }],
            temperature: None,
            max_tokens: None,
            stream: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            user: None,
            tools: None,
            tool_choice: None,
        };

        let client = OpenRouterClient::new("test-key").with_app_name("test-app");
        let built_req = client.build_request(request, Some(extras)).unwrap();

        // Verify headers
        assert!(built_req.headers().contains_key("x-title"));
        let body = String::from_utf8_lossy(built_req.body().unwrap().as_bytes().unwrap());
        assert!(body.contains("provider"));
    }
}
