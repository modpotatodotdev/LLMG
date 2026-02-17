use llmg_core::{
    provider::{ApiKeyCredentials, Credentials, LlmError, Provider},
    types::{
        ChatCompletionRequest, ChatCompletionResponse, Choice, EmbeddingRequest, EmbeddingResponse,
        Message, Usage,
    },
};

/// MiniMax API client
#[derive(Debug)]
pub struct MiniMaxClient {
    http_client: reqwest::Client,
    base_url: String,
    credentials: Box<dyn Credentials>,
}

/// MiniMax-specific request format
#[derive(Debug, serde::Serialize)]
struct MiniMaxRequest {
    model: String,
    messages: Vec<MiniMaxMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
}

/// MiniMax message format
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct MiniMaxMessage {
    role: String,
    content: String,
}

/// MiniMax response format
#[derive(Debug, serde::Deserialize)]
struct MiniMaxResponse {
    id: String,
    model: String,
    choices: Vec<MiniMaxChoice>,
    usage: Option<MiniMaxUsage>,
}

#[derive(Debug, serde::Deserialize)]
struct MiniMaxChoice {
    index: i32,
    message: MiniMaxMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct MiniMaxUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl MiniMaxClient {
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("MINIMAX_API_KEY").map_err(|_| LlmError::AuthError)?;

        Ok(Self::new(api_key))
    }

    pub fn new(api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();

        Self {
            http_client: reqwest::Client::new(),
            base_url: "https://api.minimax.ai/v1".to_string(),
            credentials: Box::new(ApiKeyCredentials::new(api_key)),
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    fn convert_request(&self, request: ChatCompletionRequest) -> MiniMaxRequest {
        let messages = request
            .messages
            .into_iter()
            .map(|msg| {
                let (role, content) = match msg {
                    Message::System { content, .. } => ("system".to_string(), content),
                    Message::User { content, .. } => ("user".to_string(), content),
                    Message::Assistant { content, .. } => {
                        ("assistant".to_string(), content.unwrap_or_default())
                    }
                    _ => ("user".to_string(), String::new()),
                };
                MiniMaxMessage { role, content }
            })
            .collect();

        MiniMaxRequest {
            model: request.model,
            messages,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
        }
    }

    fn convert_response(&self, response: MiniMaxResponse) -> ChatCompletionResponse {
        let choices = response
            .choices
            .into_iter()
            .map(|c| Choice {
                index: c.index as u32,
                message: Message::Assistant {
                    content: Some(c.message.content),
                    refusal: None,
                    tool_calls: None,
                },
                finish_reason: c.finish_reason,
            })
            .collect();

        let usage = response.usage.map(|u| Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });

        ChatCompletionResponse {
            id: response.id,
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: response.model,
            choices,
            usage,
        }
    }

    async fn make_request(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        let minimax_req = self.convert_request(request);
        let url = format!("{}/chat/completions", self.base_url);

        let mut req = self
            .http_client
            .post(&url)
            .json(&minimax_req)
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

        let minimax_resp: MiniMaxResponse = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        Ok(self.convert_response(minimax_resp))
    }
}

#[async_trait::async_trait]
impl Provider for MiniMaxClient {
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        self.make_request(request).await
    }

    async fn embeddings(&self, _request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        Err(LlmError::ProviderError(
            "Embeddings not implemented for MiniMax".to_string(),
        ))
    }
    fn provider_name(&self) -> &'static str {
        "minimax"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimax_client_creation() {
        let client = MiniMaxClient::new("test-key");
        assert_eq!(client.provider_name(), "minimax");
    }

    #[test]
    fn test_request_conversion() {
        let client = MiniMaxClient::new("test-key");

        let request = ChatCompletionRequest {
            model: "abab5.5-chat".to_string(),
            messages: vec![Message::User {
                content: "Hello!".to_string(),
                name: None,
            }],
            temperature: Some(0.8),
            max_tokens: Some(256),
            stream: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            user: None,
            tools: None,
            tool_choice: None,
        };

        let minimax_req = client.convert_request(request);

        assert_eq!(minimax_req.model, "abab5.5-chat");
        assert_eq!(minimax_req.messages.len(), 1);
        assert_eq!(minimax_req.messages[0].role, "user");
        assert_eq!(minimax_req.temperature, Some(0.8));
    }
}
