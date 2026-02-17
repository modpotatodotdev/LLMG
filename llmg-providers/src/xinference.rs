use llmg_core::{
    provider::{Credentials, LlmError, Provider},
    types::{
        ChatCompletionRequest, ChatCompletionResponse, Choice, EmbeddingRequest, EmbeddingResponse,
        Message, Usage,
    },
};

/// Xinference API client (self-hosted, no authentication)
#[derive(Debug)]
pub struct XinferenceClient {
    http_client: reqwest::Client,
    base_url: String,
    credentials: Box<dyn Credentials>,
}

/// Xinference-specific request format
#[derive(Debug, serde::Serialize)]
struct XinferenceRequest {
    model: String,
    messages: Vec<XinferenceMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
}

/// Xinference message format
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct XinferenceMessage {
    role: String,
    content: String,
}

/// Xinference response format
#[derive(Debug, serde::Deserialize)]
struct XinferenceResponse {
    id: String,
    model: String,
    choices: Vec<XinferenceChoice>,
    usage: Option<XinferenceUsage>,
}

#[derive(Debug, serde::Deserialize)]
struct XinferenceChoice {
    index: i32,
    message: XinferenceMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct XinferenceUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

/// No-op credentials for Xinference (self-hosted)
#[derive(Debug)]
struct NoCredentials;

impl Credentials for NoCredentials {
    fn apply(&self, _req: &mut reqwest::Request) -> Result<(), LlmError> {
        Ok(())
    }
}

impl XinferenceClient {
    pub fn from_env() -> Result<Self, LlmError> {
        let base_url = std::env::var("XINFERENCE_BASE_URL")
            .unwrap_or_else(|_| "http://localhost:9997".to_string());

        Ok(Self::new(base_url))
    }

    pub fn new(base_url: impl Into<String>) -> Self {
        let base_url = base_url.into();

        Self {
            http_client: reqwest::Client::new(),
            base_url,
            credentials: Box::new(NoCredentials),
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    fn convert_request(&self, request: ChatCompletionRequest) -> XinferenceRequest {
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
                XinferenceMessage { role, content }
            })
            .collect();

        XinferenceRequest {
            model: request.model,
            messages,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
        }
    }

    fn convert_response(&self, response: XinferenceResponse) -> ChatCompletionResponse {
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
        let xinference_req = self.convert_request(request);
        let url = format!("{}/v1/chat/completions", self.base_url);

        let mut req = self
            .http_client
            .post(&url)
            .json(&xinference_req)
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

        let xinference_resp: XinferenceResponse = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        Ok(self.convert_response(xinference_resp))
    }
}

#[async_trait::async_trait]
impl Provider for XinferenceClient {
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        self.make_request(request).await
    }

    async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        let url = format!("{}/v1/embeddings", self.base_url);

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
        "xinference"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xinference_client_creation() {
        let client = XinferenceClient::new("http://localhost:9997");
        assert_eq!(client.provider_name(), "xinference");
    }

    #[test]
    fn test_request_conversion() {
        let client = XinferenceClient::new("http://localhost:9997");

        let request = ChatCompletionRequest {
            model: "llama-2-7b-chat".to_string(),
            messages: vec![Message::User {
                content: "Hello!".to_string(),
                name: None,
            }],
            temperature: Some(0.7),
            max_tokens: Some(128),
            stream: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            user: None,
            tools: None,
            tool_choice: None,
        };

        let xinference_req = client.convert_request(request);

        assert_eq!(xinference_req.model, "llama-2-7b-chat");
        assert_eq!(xinference_req.messages.len(), 1);
        assert_eq!(xinference_req.messages[0].role, "user");
        assert_eq!(xinference_req.temperature, Some(0.7));
    }
}
