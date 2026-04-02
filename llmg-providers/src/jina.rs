use llmg_core::{
    provider::{ApiKeyCredentials, Credentials, LlmError, Provider},
    types::{
        ChatCompletionRequest, ChatCompletionResponse, Choice, EmbeddingRequest, EmbeddingResponse,
        Message, Usage,
    },
};

/// Jina AI API client
#[derive(Debug)]
pub struct JinaClient {
    http_client: reqwest::Client,
    base_url: String,
    credentials: Box<dyn Credentials>,
}

/// Jina AI-specific request format
#[derive(Debug, serde::Serialize)]
struct JinaRequest {
    model: String,
    messages: Vec<JinaMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
}

/// Jina AI message format
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct JinaMessage {
    role: String,
    content: String,
}

/// Jina AI response format
#[derive(Debug, serde::Deserialize)]
struct JinaResponse {
    id: String,
    model: String,
    choices: Vec<JinaChoice>,
    usage: Option<JinaUsage>,
}

#[derive(Debug, serde::Deserialize)]
struct JinaChoice {
    index: i32,
    message: JinaMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct JinaUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl JinaClient {
    /// Create a new Jina AI client from environment
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("JINAAI_API_KEY").map_err(|_| LlmError::AuthError)?;

        Ok(Self::new(api_key))
    }

    /// Create a new Jina AI client with explicit API key
    pub fn new(api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();

        Self {
            http_client: reqwest::Client::new(),
            base_url: "https://api.jina.ai/v1".to_string(),
            credentials: Box::new(ApiKeyCredentials::new(api_key)),
        }
    }

    /// Create with custom base URL
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Convert OpenAI format to Jina AI format
    fn convert_request(&self, request: ChatCompletionRequest) -> JinaRequest {
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
                JinaMessage { role, content }
            })
            .collect();

        JinaRequest {
            model: request.model,
            messages,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
        }
    }

    /// Convert Jina AI response to OpenAI format
    fn convert_response(&self, response: JinaResponse) -> ChatCompletionResponse {
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
        let jina_req = self.convert_request(request);
        let url = format!("{}/chat/completions", self.base_url);

        let mut req = self
            .http_client
            .post(&url)
            .json(&jina_req)
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

        let jina_resp: JinaResponse = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        Ok(self.convert_response(jina_resp))
    }
}

#[async_trait::async_trait]
impl Provider for JinaClient {
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
        "jina_ai"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jina_client_creation() {
        let client = JinaClient::new("test-key");
        assert_eq!(client.provider_name(), "jina_ai");
    }

    #[test]
    fn test_request_conversion() {
        let client = JinaClient::new("test-key");

        let request = ChatCompletionRequest {
            model: "jina-reranker-v1-base-en".to_string(),
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

        let jina_req = client.convert_request(request);

        assert_eq!(jina_req.model, "jina-reranker-v1-base-en");
        assert_eq!(jina_req.messages.len(), 2);
        assert_eq!(jina_req.messages[0].role, "system");
        assert_eq!(jina_req.messages[1].role, "user");
        assert_eq!(jina_req.temperature, Some(0.7));
    }
}

