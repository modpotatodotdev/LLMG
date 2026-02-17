use llmg_core::{
    provider::{ApiKeyCredentials, Credentials, LlmError, Provider},
    types::{
        ChatCompletionRequest, ChatCompletionResponse, Choice, EmbeddingRequest, EmbeddingResponse,
        Message, Usage,
    },
};
// use serde::{Deserialize, Serialize}; // removed unused imports

/// xAI Grok API client
#[derive(Debug)]
pub struct XaiClient {
    http_client: reqwest::Client,
    base_url: String,
    credentials: Box<dyn Credentials>,
}

/// xAI-specific request format
#[derive(Debug, serde::Serialize)]
struct XaiRequest {
    model: String,
    messages: Vec<XaiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
}

/// xAI message format
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct XaiMessage {
    role: String,
    content: String,
}

/// xAI response format
#[derive(Debug, serde::Deserialize)]
struct XaiResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<XaiChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<XaiUsage>,
}

#[derive(Debug, serde::Deserialize)]
struct XaiChoice {
    index: u32,
    message: XaiMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct XaiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl XaiClient {
    /// Create a new xAI client from environment
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("XAI_API_KEY").map_err(|_| LlmError::AuthError)?;

        Ok(Self::new(api_key))
    }

    /// Create a new xAI client with explicit API key
    pub fn new(api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();

        Self {
            http_client: reqwest::Client::new(),
            base_url: "https://api.x.ai/v1".to_string(),
            credentials: Box::new(ApiKeyCredentials::new(api_key)),
        }
    }

    /// Create with custom base URL
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Convert OpenAI format to xAI format
    fn convert_request(&self, request: ChatCompletionRequest) -> XaiRequest {
        let messages = request
            .messages
            .into_iter()
            .filter_map(|msg| match msg {
                Message::System { content, .. } => Some(XaiMessage {
                    role: "system".to_string(),
                    content,
                }),
                Message::User { content, .. } => Some(XaiMessage {
                    role: "user".to_string(),
                    content,
                }),
                Message::Assistant { content, .. } => content.map(|c| XaiMessage {
                    role: "assistant".to_string(),
                    content: c,
                }),
                _ => None,
            })
            .collect();

        XaiRequest {
            model: request.model,
            messages,
            stream: request.stream,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            top_p: request.top_p,
        }
    }

    /// Convert xAI response to OpenAI format
    fn convert_response(&self, response: XaiResponse) -> ChatCompletionResponse {
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
        let xai_req = self.convert_request(request);
        let url = format!("{}/chat/completions", self.base_url);

        let mut req = self
            .http_client
            .post(&url)
            .json(&xai_req)
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

        let xai_resp: XaiResponse = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        Ok(self.convert_response(xai_resp))
    }
}

#[async_trait::async_trait]
impl Provider for XaiClient {
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        self.make_request(request).await
    }

    async fn embeddings(&self, _request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        Err(LlmError::ProviderError(
            "xAI does not support embeddings".to_string(),
        ))
    }
    fn provider_name(&self) -> &'static str {
        "xai"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xai_client_creation() {
        let client = XaiClient::new("test-key");
        assert_eq!(client.provider_name(), "xai");
    }

    #[test]
    fn test_request_conversion() {
        let client = XaiClient::new("test-key");

        let request = ChatCompletionRequest {
            model: "grok-beta".to_string(),
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
        };

        let xai_req = client.convert_request(request);

        assert_eq!(xai_req.model, "grok-beta");
        assert_eq!(xai_req.messages.len(), 2);
        assert_eq!(xai_req.messages[0].role, "system");
        assert_eq!(xai_req.messages[1].role, "user");
    }
}
