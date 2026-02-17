//! Request routing for LLMG Gateway
//!
//! Parses provider/model format and routes to appropriate provider.

use crate::streaming::SseStream;
use crate::GatewayState;
use axum::{
    extract::Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use llmg_core::types::ChatCompletionRequest;
use serde_json::json;
use std::sync::Arc;

/// We now use the implementation in llmg-core::provider::parse_model_id
pub use llmg_core::provider::parse_model_id as core_parse_model_id;

/// Parse a model identifier in the format "provider/model"
/// Wrapper around core `parse_model_id` that maps to `RoutingError`
pub fn parse_model_id(model_id: &str) -> Result<(&str, String), RoutingError> {
    core_parse_model_id(model_id).map_err(RoutingError::InvalidFormat)
}

/// Extract provider from chat completion request
#[allow(dead_code)]
pub fn extract_provider_from_request(
    request: &ChatCompletionRequest,
) -> Result<String, RoutingError> {
    parse_model_id(&request.model).map(|(provider, _)| provider.to_string())
}

/// Error types for routing
#[derive(Debug)]
pub enum RoutingError {
    InvalidFormat(String),
    UnknownProvider(String),
}

impl IntoResponse for RoutingError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            RoutingError::InvalidFormat(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            RoutingError::UnknownProvider(provider) => (
                StatusCode::NOT_FOUND,
                format!("Unknown provider: {}", provider),
            ),
        };

        let body = json!({
            "error": {
                "message": message,
                "type": "invalid_request_error",
            }
        });

        (status, Json(body)).into_response()
    }
}

/// Model alias mapping
/// Allows users to use short names that map to full provider/model paths
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ModelAliases {
    aliases: std::collections::HashMap<String, String>,
}

impl ModelAliases {
    /// Create a new empty alias map
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            aliases: std::collections::HashMap::new(),
        }
    }

    /// Register an alias
    #[allow(dead_code)]
    pub fn register(&mut self, alias: impl Into<String>, full_model: impl Into<String>) {
        self.aliases.insert(alias.into(), full_model.into());
    }

    /// Resolve an alias to full model path
    #[allow(dead_code)]
    pub fn resolve(&self, model: &str) -> String {
        self.aliases
            .get(model)
            .cloned()
            .unwrap_or_else(|| model.to_string())
    }

    /// Create with common aliases
    #[allow(dead_code)]
    pub fn with_defaults() -> Self {
        let mut aliases = Self::new();

        aliases.register("gpt-4", "openai/gpt-4");
        aliases.register("gpt-4-turbo", "openai/gpt-4-turbo");
        aliases.register("gpt-4o", "openai/gpt-4o");
        aliases.register("gpt-4o-mini", "openai/gpt-4o-mini");
        aliases.register("gpt-3.5", "openai/gpt-3.5-turbo");

        aliases.register("claude", "anthropic/claude-3-opus-20240229");
        aliases.register("claude-opus", "anthropic/claude-3-opus-20240229");
        aliases.register("claude-sonnet", "anthropic/claude-3-sonnet-20240229");
        aliases.register("claude-3", "anthropic/claude-3-opus-20240229");
        aliases.register("claude-3-opus", "anthropic/claude-3-opus-20240229");
        aliases.register("claude-3-sonnet", "anthropic/claude-3-sonnet-20240229");
        aliases.register("claude-3-haiku", "anthropic/claude-3-haiku-20240307");

        aliases.register("gemini", "antigravity/gemini-2.0-flash");
        aliases.register("gemini-pro", "antigravity/gemini-1.5-pro");
        aliases.register("gemini-flash", "antigravity/gemini-2.0-flash");
        aliases.register("gemini-2", "antigravity/gemini-2.0-pro-exp-02-05");
        aliases.register("gemini-1.5", "antigravity/gemini-1.5-pro");

        aliases.register("llama", "meta_llama/llama-3.1-70b-instruct");
        aliases.register("llama-3", "meta_llama/llama-3.1-70b-instruct");
        aliases.register("llama-2", "meta_llama/llama-2-70b-chat-hf");
        aliases.register("llama-3.1", "meta_llama/llama-3.1-70b-instruct");
        aliases.register("llama-3.2", "meta_llama/llama-3.2-90b-vision-instruct");

        aliases.register("mistral", "mistral/mistral-large");
        aliases.register("mistral-large", "mistral/mistral-large");
        aliases.register("mistral-medium", "mistral/mistral-medium");
        aliases.register("mistral-small", "mistral/mistral-small");
        aliases.register("mixtral", "mistral/mixtral-8x7b");
        aliases.register("codestral", "mistral/codestral");

        aliases.register("deepseek", "deepseek/deepseek-chat");
        aliases.register("deepseek-chat", "deepseek/deepseek-chat");
        aliases.register("deepseek-coder", "deepseek/deepseek-coder");

        aliases.register("groq", "groq/llama3-70b-8192");
        aliases.register("llama3-groq", "groq/llama3-70b-8192");
        aliases.register("mixtral-groq", "groq/mixtral-8x7b-32768");

        aliases.register("command", "cohere/command");
        aliases.register("command-r", "cohere/command-r");
        aliases.register("command-r-plus", "cohere/command-r-plus");

        aliases.register("ollama-llama3", "ollama/llama3");
        aliases.register("ollama-mistral", "ollama/mistral");
        aliases.register("ollama-codellama", "ollama/codellama");

        aliases.register("bedrock", "bedrock/anthropic.claude-3-opus-20240229-v1:0");

        aliases.register("vertex", "vertex_ai/gemini-1.5-pro");

        aliases.register("hf", "huggingface/gpt2");

        aliases.register("perplexity", "perplexity/pplx-7b-online");

        aliases.register("grok", "xai/grok-beta");
        aliases.register("xai", "xai/grok-beta");

        aliases.register("qwen", "openrouter/qwen/qwen-2.5-72b-instruct");

        aliases.register("hermes", "openrouter/nousresearch/hermes-3-llama-3.1-405b");

        aliases
    }
}

impl Default for ModelAliases {
    fn default() -> Self {
        Self::new()
    }
}

/// Route request to appropriate provider
/// Route request to appropriate provider
pub async fn route_chat_completion(
    state: Arc<GatewayState>,
    mut request: ChatCompletionRequest,
) -> Result<Response, RoutingError> {
    // Parse the model ID
    let (provider_name, model_name) = parse_model_id(&request.model)?;
    let provider_name = provider_name.to_string();

    // Update model name to remove provider prefix for the actual provider call
    request.model = model_name;

    // Look up provider in registry
    let provider = state
        .registry
        .get(&provider_name)
        .ok_or(RoutingError::UnknownProvider(provider_name))?;

    // Perform the completion with retry logic
    use crate::middleware::{with_retry, RetryConfig};
    let retry_config = RetryConfig::default();

    match with_retry(&retry_config, || {
        let provider = provider.clone();
        let request = request.clone();
        async move { provider.chat_completion(request).await }
    })
    .await
    {
        Ok(response) => Ok((StatusCode::OK, Json(response)).into_response()),
        Err(err) => {
            // Convert LlmError to something OpenAI-compatible
            let status = match &err {
                llmg_core::provider::LlmError::RateLimitError => StatusCode::TOO_MANY_REQUESTS,
                llmg_core::provider::LlmError::AuthError => StatusCode::UNAUTHORIZED,
                llmg_core::provider::LlmError::InvalidRequest(_) => StatusCode::BAD_REQUEST,
                llmg_core::provider::LlmError::NotFound => StatusCode::NOT_FOUND,
                llmg_core::provider::LlmError::InternalError(_) => {
                    StatusCode::INTERNAL_SERVER_ERROR
                }
                llmg_core::provider::LlmError::Timeout => StatusCode::GATEWAY_TIMEOUT,
                llmg_core::provider::LlmError::UnsupportedFeature => StatusCode::NOT_IMPLEMENTED,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            };

            let body = json!({
                "error": {
                    "message": err.to_string(),
                    "type": "provider_error",
                }
            });

            Ok((status, Json(body)).into_response())
        }
    }
}

/// Route streaming request to appropriate provider
pub async fn route_chat_completion_stream(
    state: Arc<GatewayState>,
    mut request: ChatCompletionRequest,
) -> Result<Response, RoutingError> {
    let (provider_name, model_name) = parse_model_id(&request.model)?;
    let provider_name = provider_name.to_string();

    request.model = model_name;

    let provider = state
        .registry
        .get(&provider_name)
        .ok_or(RoutingError::UnknownProvider(provider_name))?;

    match provider.chat_completion_stream(request).await {
        Ok(stream) => {
            let sse_stream = SseStream::from_provider_stream(stream);
            Ok(sse_stream.into_response())
        }
        Err(err) => {
            let status = match &err {
                llmg_core::provider::LlmError::RateLimitError => StatusCode::TOO_MANY_REQUESTS,
                llmg_core::provider::LlmError::AuthError => StatusCode::UNAUTHORIZED,
                llmg_core::provider::LlmError::InvalidRequest(_) => StatusCode::BAD_REQUEST,
                llmg_core::provider::LlmError::NotFound => StatusCode::NOT_FOUND,
                llmg_core::provider::LlmError::InternalError(_) => {
                    StatusCode::INTERNAL_SERVER_ERROR
                }
                llmg_core::provider::LlmError::Timeout => StatusCode::GATEWAY_TIMEOUT,
                llmg_core::provider::LlmError::UnsupportedFeature => StatusCode::NOT_IMPLEMENTED,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            };

            let body = json!({
                "error": {
                    "message": err.to_string(),
                    "type": "provider_error",
                }
            });

            Ok((status, Json(body)).into_response())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_model_aliases() {
        let aliases = ModelAliases::with_defaults();

        assert_eq!(aliases.resolve("gpt-4"), "openai/gpt-4");
        assert_eq!(aliases.resolve("openai/gpt-4"), "openai/gpt-4"); // Passthrough
        assert_eq!(aliases.resolve("unknown-model"), "unknown-model"); // Unknown passes through
    }
}
