use llmg_core::{
    provider::{ApiKeyCredentials, Credentials, LlmError, Provider},
    types::{ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse},
};

/// Groq API client
#[derive(Debug)]
pub struct GroqClient {
    http_client: reqwest::Client,
    base_url: String,
    credentials: Box<dyn Credentials>,
}

impl GroqClient {
    /// Create a new Groq client from environment
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("GROQ_API_KEY").map_err(|_| LlmError::AuthError)?;

        Ok(Self::new(api_key))
    }

    /// Create a new Groq client with explicit API key
    pub fn new(api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();
        let credentials = Box::new(ApiKeyCredentials::bearer(api_key));

        Self {
            http_client: reqwest::Client::new(),
            base_url: "https://api.groq.com/openai/v1".to_string(),
            credentials,
        }
    }

    /// Create with custom base URL (for proxies)
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
impl Provider for GroqClient {
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        self.make_request(request).await
    }

    async fn embeddings(&self, _request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        Err(LlmError::ProviderError(
            "Groq does not support embeddings".to_string(),
        ))
    }
    fn provider_name(&self) -> &'static str {
        "groq"
    }

    fn supported_models(&self) -> Vec<String> {
        vec![
            "llama-3.3-70b-versatile".to_string(),
            "llama-3.3-70b-specdec".to_string(),
            "llama3-70b-8192".to_string(),
            "llama3-8b-8192".to_string(),
            "llama3-groq-70b-8192-tool-use-preview".to_string(),
            "llama3-groq-8b-8192-tool-use-preview".to_string(),
            "llama-3.1-8b-instant".to_string(),
            "llama-3.2-1b-preview".to_string(),
            "llama-3.2-3b-preview".to_string(),
            "llama-3.2-11b-vision-preview".to_string(),
            "llama-3.2-90b-vision-preview".to_string(),
            "mixtral-8x7b-32768".to_string(),
            "gemma2-9b-it".to_string(),
            "deepseek-r1-distill-llama-70b".to_string(),
            "qwen-2.5-32b".to_string(),
            "qwen-qwq-32b-preview".to_string(),
            "whisper-large-v3".to_string(),
            "whisper-large-v3-turbo".to_string(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_groq_client_creation() {
        let client = GroqClient::new("test-key");
        assert_eq!(client.provider_name(), "groq");
    }

    #[test]
    fn test_from_env_missing_key() {
        // Temporarily remove env var
        let original = std::env::var("GROQ_API_KEY").ok();
        std::env::remove_var("GROQ_API_KEY");

        let result = GroqClient::from_env();
        assert!(result.is_err());

        // Restore
        if let Some(key) = original {
            std::env::set_var("GROQ_API_KEY", key);
        }
    }
}
