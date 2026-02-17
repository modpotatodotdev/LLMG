use llmg_core::{
    provider::{ApiKeyCredentials, Credentials, LlmError, Provider},
    types::{
        ChatCompletionRequest, ChatCompletionResponse, Choice, EmbeddingRequest, EmbeddingResponse,
        Message, Usage,
    },
};
// use serde::{Serialize, Deserialize};

/// DeepSeek API client
#[derive(Debug)]
pub struct DeepseekClient {
    http_client: reqwest::Client,
    base_url: String,
    credentials: Box<dyn Credentials>,
}

/// DeepSeek-specific request format
#[derive(Debug, serde::Serialize)]
struct DeepseekRequest {
    model: String,
    messages: Vec<DeepseekMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

/// DeepSeek message format
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct DeepseekMessage {
    role: String,
    content: String,
}

/// DeepSeek response format
#[derive(Debug, serde::Deserialize)]
struct DeepseekResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<DeepseekChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<DeepseekUsage>,
}

#[derive(Debug, serde::Deserialize)]
struct DeepseekChoice {
    index: u32,
    message: DeepseekMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct DeepseekUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl DeepseekClient {
    /// Create a new DeepSeek client from environment
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("DEEPSEEK_API_KEY").map_err(|_| LlmError::AuthError)?;

        Ok(Self::new(api_key))
    }

    /// Create a new DeepSeek client with explicit API key
    pub fn new(api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();

        Self {
            http_client: reqwest::Client::new(),
            base_url: "https://api.deepseek.com".to_string(),
            credentials: Box::new(ApiKeyCredentials::new(api_key)),
        }
    }

    /// Create with custom base URL
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Convert OpenAI format to DeepSeek format
    fn convert_request(&self, request: ChatCompletionRequest) -> DeepseekRequest {
        let messages = request
            .messages
            .into_iter()
            .filter_map(|msg| match msg {
                Message::System { content, .. } => Some(DeepseekMessage {
                    role: "system".to_string(),
                    content,
                }),
                Message::User { content, .. } => Some(DeepseekMessage {
                    role: "user".to_string(),
                    content,
                }),
                Message::Assistant { content, .. } => content.map(|c| DeepseekMessage {
                    role: "assistant".to_string(),
                    content: c,
                }),
                _ => None,
            })
            .collect();

        DeepseekRequest {
            model: request.model,
            messages,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            top_p: request.top_p,
            stop: request.stop,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            stream: request.stream,
        }
    }

    /// Convert DeepSeek response to OpenAI format
    fn convert_response(&self, response: DeepseekResponse) -> ChatCompletionResponse {
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
        let deepseek_req = self.convert_request(request);
        let url = format!("{}/chat/completions", self.base_url);

        let mut req = self
            .http_client
            .post(&url)
            .json(&deepseek_req)
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

        let deepseek_resp: DeepseekResponse = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        Ok(self.convert_response(deepseek_resp))
    }
}

#[async_trait::async_trait]
impl Provider for DeepseekClient {
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        self.make_request(request).await
    }

    async fn embeddings(&self, _request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        Err(LlmError::ProviderError(
            "DeepSeek does not support embeddings".to_string(),
        ))
    }
    fn provider_name(&self) -> &'static str {
        "deepseek"
    }

    fn supported_models(&self) -> Vec<String> {
        vec![
            "deepseek-chat".to_string(),
            "deepseek-coder".to_string(),
            "deepseek-reasoner".to_string(),
            "deepseek-r1".to_string(),
            "deepseek-v3".to_string(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deepseek_client_creation() {
        let client = DeepseekClient::new("test-key");
        assert_eq!(client.provider_name(), "deepseek");
    }

    #[test]
    fn test_request_conversion() {
        let client = DeepseekClient::new("test-key");

        let request = ChatCompletionRequest {
            model: "deepseek-chat".to_string(),
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

        let deepseek_req = client.convert_request(request);

        assert_eq!(deepseek_req.model, "deepseek-chat");
        assert_eq!(deepseek_req.messages.len(), 1);
        assert_eq!(deepseek_req.messages[0].role, "user");
    }
}
