use llmg_core::{
    provider::{ApiKeyCredentials, Credentials, LlmError, Provider},
    types::{
        ChatCompletionRequest, ChatCompletionResponse, Choice, EmbeddingRequest, EmbeddingResponse,
        Message, Usage,
    },
};

/// Helicone API client for proxy/gateway
#[derive(Debug)]
pub struct HeliconeClient {
    http_client: reqwest::Client,
    base_url: String,
    credentials: Box<dyn Credentials>,
}

#[derive(Debug, serde::Serialize)]
struct HeliconeRequest {
    model: String,
    messages: Vec<HeliconeMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct HeliconeMessage {
    role: String,
    content: String,
}

#[derive(Debug, serde::Deserialize)]
struct HeliconeResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<HeliconeChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<HeliconeUsage>,
}

#[derive(Debug, serde::Deserialize)]
struct HeliconeChoice {
    index: u32,
    message: HeliconeMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct HeliconeUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl HeliconeClient {
    /// Create a new Helicone client from environment
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("HELICONE_API_KEY").map_err(|_| LlmError::AuthError)?;

        Ok(Self::new(api_key))
    }

    /// Create a new Helicone client with explicit API key
    pub fn new(api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();

        Self {
            http_client: reqwest::Client::new(),
            base_url: "https://oai.helicone.ai/v1".to_string(),
            credentials: Box::new(ApiKeyCredentials::new(api_key)),
        }
    }

    /// Create with custom base URL
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    fn convert_request(&self, request: ChatCompletionRequest) -> HeliconeRequest {
        let messages = request
            .messages
            .into_iter()
            .filter_map(|msg| match msg {
                Message::System { content, .. } => Some(HeliconeMessage {
                    role: "system".to_string(),
                    content,
                }),
                Message::User { content, .. } => Some(HeliconeMessage {
                    role: "user".to_string(),
                    content,
                }),
                Message::Assistant { content, .. } => content.map(|c| HeliconeMessage {
                    role: "assistant".to_string(),
                    content: c,
                }),
                _ => None,
            })
            .collect();

        HeliconeRequest {
            model: request.model,
            messages,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            top_p: request.top_p,
            stream: request.stream,
        }
    }

    fn convert_response(&self, response: HeliconeResponse) -> ChatCompletionResponse {
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
        let helicone_req = self.convert_request(request);
        let url = format!("{}/chat/completions", self.base_url);

        let mut req = self
            .http_client
            .post(&url)
            .json(&helicone_req)
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

        let helicone_resp: HeliconeResponse = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        Ok(self.convert_response(helicone_resp))
    }
}

#[async_trait::async_trait]
impl Provider for HeliconeClient {
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        self.make_request(request).await
    }

    async fn embeddings(&self, _request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        Err(LlmError::ProviderError(
            "Helicone does not support embeddings directly".to_string(),
        ))
    }
    fn provider_name(&self) -> &'static str {
        "helicone"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_helicone_client_creation() {
        let client = HeliconeClient::new("test-key");
        assert_eq!(client.provider_name(), "helicone");
    }

    #[test]
    fn test_request_conversion() {
        let client = HeliconeClient::new("test-key");

        let request = ChatCompletionRequest {
            model: "gpt-4".to_string(),
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
            response_format: None,
        };

        let helicone_req = client.convert_request(request);

        assert_eq!(helicone_req.model, "gpt-4");
        assert_eq!(helicone_req.messages.len(), 1);
        assert_eq!(helicone_req.messages[0].role, "user");
    }
}

