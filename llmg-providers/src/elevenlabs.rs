use llmg_core::{
    provider::{ApiKeyCredentials, Credentials, LlmError, Provider},
    types::{ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse},
};

/// ElevenLabs API client (Audio)
#[derive(Debug)]
pub struct ElevenLabsClient {
    http_client: reqwest::Client,
    base_url: String,
    credentials: Box<dyn Credentials>,
}

impl ElevenLabsClient {
    /// Create a new ElevenLabs client from environment
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("ELEVENLABS_API_KEY").map_err(|_| LlmError::AuthError)?;
        Ok(Self::new(api_key))
    }

    pub fn new(api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();
        Self {
            http_client: reqwest::Client::new(),
            base_url: "https://api.elevenlabs.io/v1".to_string(),
            credentials: Box::new(ApiKeyCredentials::with_header(api_key, "xi-api-key")),
        }
    }
}

#[async_trait::async_trait]
impl Provider for ElevenLabsClient {
    async fn chat_completion(
        &self,
        _request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        // ElevenLabs is primarily TTS. We'll map chat completion to TTS if it makes sense,
        // or just return an error for now.
        // Parity with LiteLLM for ElevenLabs usually means their TTS API.
        Err(LlmError::UnsupportedFeature)
    }

    async fn embeddings(&self, _request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        Err(LlmError::UnsupportedFeature)
    }
    fn provider_name(&self) -> &'static str {
        "elevenlabs"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elevenlabs_client_creation() {
        let client = ElevenLabsClient::new("test-key");
        assert_eq!(client.provider_name(), "elevenlabs");
    }

    #[test]
    fn test_from_env_missing_key() {
        let original = std::env::var("ELEVENLABS_API_KEY").ok();
        std::env::remove_var("ELEVENLABS_API_KEY");
        let result = ElevenLabsClient::from_env();
        assert!(result.is_err());
        if let Some(key) = original {
            std::env::set_var("ELEVENLABS_API_KEY", key);
        }
    }
}
