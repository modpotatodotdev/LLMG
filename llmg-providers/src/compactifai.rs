use llmg_core::{
    provider::{ApiKeyCredentials, Credentials, LlmError, Provider},
    types::{
        ChatCompletionRequest, ChatCompletionResponse, Choice, EmbeddingRequest, EmbeddingResponse,
        Message, Usage,
    },
};

/// CompactifAI API client
#[derive(Debug)]
pub struct CompactifAiClient {
    http_client: reqwest::Client,
    base_url: String,
    credentials: Box<dyn Credentials>,
}

/// CompactifAI-specific request format
#[derive(Debug, serde::Serialize)]
struct CompactifAiRequest {
    model: String,
    messages: Vec<CompactifAiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

/// CompactifAI message format
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct CompactifAiMessage {
    role: String,
    content: String,
}

/// CompactifAI response format
#[derive(Debug, serde::Deserialize)]
struct CompactifAiResponse {
    id: String,
    choices: Vec<CompactifAiChoice>,
    model: String,
    usage: CompactifAiUsage,
}

#[derive(Debug, serde::Deserialize)]
struct CompactifAiChoice {
    index: u32,
    message: CompactifAiMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct CompactifAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl CompactifAiClient {
    /// Create a new CompactifAI client from environment
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("COMPACTIFAI_API_KEY").map_err(|_| LlmError::AuthError)?;

        Ok(Self::new(api_key))
    }

    /// Create a new CompactifAI client with explicit API key
    pub fn new(api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();

        Self {
            http_client: reqwest::Client::new(),
            base_url: "https://api.compactif.ai/v1".to_string(),
            credentials: Box::new(ApiKeyCredentials::with_header(api_key, "Authorization")),
        }
    }

    /// Create with custom base URL
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Convert OpenAI format to CompactifAI format
    fn convert_request(&self, request: ChatCompletionRequest) -> CompactifAiRequest {
        let messages = request
            .messages
            .into_iter()
            .filter_map(|msg| match msg {
                Message::System { content, .. } => Some(CompactifAiMessage {
                    role: "system".to_string(),
                    content,
                }),
                Message::User { content, .. } => Some(CompactifAiMessage {
                    role: "user".to_string(),
                    content,
                }),
                Message::Assistant { content, .. } => content.map(|content| CompactifAiMessage {
                    role: "assistant".to_string(),
                    content,
                }),
                _ => None,
            })
            .collect();

        CompactifAiRequest {
            model: request.model,
            messages,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            stream: request.stream,
        }
    }

    /// Convert CompactifAI response to OpenAI format
    fn convert_response(&self, response: CompactifAiResponse) -> ChatCompletionResponse {
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
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: response.model,
            choices,
            usage: Some(Usage {
                prompt_tokens: response.usage.prompt_tokens,
                completion_tokens: response.usage.completion_tokens,
                total_tokens: response.usage.total_tokens,
            }),
        }
    }

    async fn make_request(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        let compactifai_req = self.convert_request(request);
        let url = format!("{}/chat/completions", self.base_url);

        let mut req = self
            .http_client
            .post(&url)
            .json(&compactifai_req)
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

        let compactifai_resp: CompactifAiResponse = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        Ok(self.convert_response(compactifai_resp))
    }
}

#[async_trait::async_trait]
impl Provider for CompactifAiClient {
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        self.make_request(request).await
    }

    async fn embeddings(&self, _request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        Err(LlmError::ProviderError(
            "CompactifAI does not support embeddings".to_string(),
        ))
    }
    fn provider_name(&self) -> &'static str {
        "compactifai"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compactifai_client_creation() {
        let client = CompactifAiClient::new("test-key");
        assert_eq!(client.provider_name(), "compactifai");
    }

    #[test]
    fn test_request_conversion() {
        let client = CompactifAiClient::new("test-key");

        let request = ChatCompletionRequest {
            model: "compactifai-1".to_string(),
            messages: vec![
                Message::System {
                    content: "You are a helpful assistant".to_string(),
                    name: None,
                },
                Message::User {
                    content: "Hello!".to_string(),
                    name: None,
                },
            ],
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
            response_format: None,
        };

        let compactifai_req = client.convert_request(request);

        assert_eq!(compactifai_req.model, "compactifai-1");
        assert_eq!(compactifai_req.messages.len(), 2);
        assert_eq!(compactifai_req.messages[0].role, "system");
        assert_eq!(compactifai_req.messages[1].role, "user");
        assert_eq!(compactifai_req.temperature, Some(0.7));
        assert_eq!(compactifai_req.max_tokens, Some(100));
    }
}

