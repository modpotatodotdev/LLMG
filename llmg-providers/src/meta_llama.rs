use llmg_core::{
    provider::{ApiKeyCredentials, Credentials, LlmError, Provider},
    types::{
        ChatCompletionRequest, ChatCompletionResponse, Choice, EmbeddingRequest, EmbeddingResponse,
        Message, Usage,
    },
};

/// Meta Llama API client
#[derive(Debug)]
pub struct MetaLlamaClient {
    http_client: reqwest::Client,
    base_url: String,
    credentials: Box<dyn Credentials>,
}

/// Meta Llama-specific request format
#[derive(Debug, serde::Serialize)]
struct MetaLlamaRequest {
    model: String,
    messages: Vec<MetaLlamaMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
}

/// Meta Llama message format
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct MetaLlamaMessage {
    role: String,
    content: String,
}

/// Meta Llama response format
#[derive(Debug, serde::Deserialize)]
struct MetaLlamaResponse {
    id: String,
    model: String,
    choices: Vec<MetaLlamaChoice>,
    usage: Option<MetaLlamaUsage>,
}

#[derive(Debug, serde::Deserialize)]
struct MetaLlamaChoice {
    index: i32,
    message: MetaLlamaMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct MetaLlamaUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl MetaLlamaClient {
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("META_API_KEY").map_err(|_| LlmError::AuthError)?;

        Ok(Self::new(api_key))
    }

    pub fn new(api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();

        Self {
            http_client: reqwest::Client::new(),
            base_url: "https://api.meta.com/v1".to_string(),
            credentials: Box::new(ApiKeyCredentials::new(api_key)),
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    fn convert_request(&self, request: ChatCompletionRequest) -> MetaLlamaRequest {
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
                MetaLlamaMessage { role, content }
            })
            .collect();

        MetaLlamaRequest {
            model: request.model,
            messages,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
        }
    }

    fn convert_response(&self, response: MetaLlamaResponse) -> ChatCompletionResponse {
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
        let llama_req = self.convert_request(request);
        let url = format!("{}/chat/completions", self.base_url);

        let mut req = self
            .http_client
            .post(&url)
            .json(&llama_req)
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

        let llama_resp: MetaLlamaResponse = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        Ok(self.convert_response(llama_resp))
    }
}

#[async_trait::async_trait]
impl Provider for MetaLlamaClient {
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        self.make_request(request).await
    }

    async fn embeddings(&self, _request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        Err(LlmError::ProviderError(
            "Embeddings not implemented for Meta Llama".to_string(),
        ))
    }
    fn provider_name(&self) -> &'static str {
        "meta_llama"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_llama_client_creation() {
        let client = MetaLlamaClient::new("test-key");
        assert_eq!(client.provider_name(), "meta_llama");
    }

    #[test]
    fn test_request_conversion() {
        let client = MetaLlamaClient::new("test-key");

        let request = ChatCompletionRequest {
            model: "llama-3-70b".to_string(),
            messages: vec![Message::User {
                content: "Hello!".to_string(),
                name: None,
            }],
            temperature: Some(0.6),
            max_tokens: Some(512),
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

        let llama_req = client.convert_request(request);

        assert_eq!(llama_req.model, "llama-3-70b");
        assert_eq!(llama_req.messages.len(), 1);
        assert_eq!(llama_req.messages[0].role, "user");
        assert_eq!(llama_req.temperature, Some(0.6));
    }
}

