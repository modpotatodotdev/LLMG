use llmg_core::{
    provider::{ApiKeyCredentials, Credentials, LlmError, Provider},
    types::{
        ChatCompletionRequest, ChatCompletionResponse, Choice, EmbeddingRequest, EmbeddingResponse,
        Message, Usage,
    },
};

/// Perplexity API client
#[derive(Debug)]
pub struct PerplexityClient {
    http_client: reqwest::Client,
    base_url: String,
    credentials: Box<dyn Credentials>,
}

/// Perplexity-specific request format
#[derive(Debug, serde::Serialize)]
struct PerplexityRequest {
    model: String,
    messages: Vec<PerplexityMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

/// Perplexity message format
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct PerplexityMessage {
    role: String,
    content: String,
}

/// Perplexity response format
#[derive(Debug, serde::Deserialize)]
struct PerplexityResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<PerplexityChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<PerplexityUsage>,
}

#[derive(Debug, serde::Deserialize)]
struct PerplexityChoice {
    index: u32,
    message: PerplexityMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct PerplexityUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl PerplexityClient {
    /// Create a new Perplexity client from environment
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("PERPLEXITY_API_KEY").map_err(|_| LlmError::AuthError)?;

        Ok(Self::new(api_key))
    }

    /// Create a new Perplexity client with explicit API key
    pub fn new(api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();

        Self {
            http_client: reqwest::Client::new(),
            base_url: "https://api.perplexity.ai/v1".to_string(),
            credentials: Box::new(ApiKeyCredentials::new(api_key)),
        }
    }

    /// Create with custom base URL
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Convert OpenAI format to Perplexity format
    fn convert_request(&self, request: ChatCompletionRequest) -> PerplexityRequest {
        let messages = request
            .messages
            .into_iter()
            .filter_map(|msg| match msg {
                Message::System { content, .. } => Some(PerplexityMessage {
                    role: "system".to_string(),
                    content,
                }),
                Message::User { content, .. } => Some(PerplexityMessage {
                    role: "user".to_string(),
                    content,
                }),
                Message::Assistant { content, .. } => content.map(|c| PerplexityMessage {
                    role: "assistant".to_string(),
                    content: c,
                }),
                _ => None,
            })
            .collect();

        PerplexityRequest {
            model: request.model,
            messages,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            top_p: request.top_p,
            stream: request.stream,
        }
    }

    /// Convert Perplexity response to OpenAI format
    fn convert_response(&self, response: PerplexityResponse) -> ChatCompletionResponse {
        let choices = response
            .choices
            .into_iter()
            .map(|choice| Choice {
                index: choice.index,
                message: Message::Assistant {
                    content: Some(choice.message.content),
                    refusal: None,
                    tool_calls: None,
                },
                finish_reason: choice.finish_reason,
            })
            .collect();

        ChatCompletionResponse {
            id: response.id,
            object: response.object,
            created: response.created as i64,
            model: response.model,
            choices,
            usage: response.usage.map(|u| Usage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
            }),
        }
    }

    async fn make_request(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        let perplexity_req = self.convert_request(request);
        let url = format!("{}/chat/completions", self.base_url);

        let mut req = self
            .http_client
            .post(&url)
            .json(&perplexity_req)
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

        let perplexity_resp: PerplexityResponse = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        Ok(self.convert_response(perplexity_resp))
    }
}

#[async_trait::async_trait]
impl Provider for PerplexityClient {
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        self.make_request(request).await
    }

    async fn embeddings(&self, _request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        Err(LlmError::ProviderError(
            "Perplexity does not support embeddings".to_string(),
        ))
    }
    fn provider_name(&self) -> &'static str {
        "perplexity"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perplexity_client_creation() {
        let client = PerplexityClient::new("test-key");
        assert_eq!(client.provider_name(), "perplexity");
    }

    #[test]
    fn test_request_conversion() {
        let client = PerplexityClient::new("test-key");

        let request = ChatCompletionRequest {
            model: "sonar-pro".to_string(),
            messages: vec![Message::User {
                content: "Hello!".to_string(),
                name: None,
            }],
            temperature: Some(0.7),
            max_tokens: Some(100),
            stream: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            user: None,
            tools: None,
            tool_choice: None,
        };

        let perplexity_req = client.convert_request(request);

        assert_eq!(perplexity_req.model, "sonar-pro");
        assert_eq!(perplexity_req.messages.len(), 1);
        assert_eq!(perplexity_req.messages[0].role, "user");
    }
}
