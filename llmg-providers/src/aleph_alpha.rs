use llmg_core::{
    provider::{ApiKeyCredentials, Credentials, LlmError, Provider},
    types::{
        ChatCompletionRequest, ChatCompletionResponse, Choice, EmbeddingRequest, EmbeddingResponse,
        Message, Usage,
    },
};

/// Aleph Alpha API client
#[derive(Debug)]
pub struct AlephAlphaClient {
    http_client: reqwest::Client,
    base_url: String,
    credentials: Box<dyn Credentials>,
}

/// Aleph Alpha-specific request format
#[derive(Debug, serde::Serialize)]
struct AlephAlphaRequest {
    model: String,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
}

/// Aleph Alpha response format
#[derive(Debug, serde::Deserialize)]
struct AlephAlphaResponse {
    id: String,
    model: String,
    choices: Vec<AlephAlphaChoice>,
    usage: Option<AlephAlphaUsage>,
}

#[derive(Debug, serde::Deserialize)]
struct AlephAlphaChoice {
    index: i32,
    text: String,
    finish_reason: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct AlephAlphaUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl AlephAlphaClient {
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("ALEPHALPHA_API_KEY").map_err(|_| LlmError::AuthError)?;

        Ok(Self::new(api_key))
    }

    pub fn new(api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();

        Self {
            http_client: reqwest::Client::new(),
            base_url: "https://api.aleph-alpha.com/v1".to_string(),
            credentials: Box::new(ApiKeyCredentials::new(api_key)),
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    fn convert_request(&self, request: ChatCompletionRequest) -> AlephAlphaRequest {
        let prompt = request
            .messages
            .into_iter()
            .map(|msg| match msg {
                Message::System { content, .. } => format!("System: {}\n", content),
                Message::User { content, .. } => format!("User: {}\n", content),
                Message::Assistant { content, .. } => {
                    format!("Assistant: {}\n", content.unwrap_or_default())
                }
                _ => String::new(),
            })
            .collect::<Vec<_>>()
            .concat();

        AlephAlphaRequest {
            model: request.model,
            prompt,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: None,
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
        }
    }

    fn convert_response(&self, response: AlephAlphaResponse) -> ChatCompletionResponse {
        let choices = response
            .choices
            .into_iter()
            .map(|c| Choice {
                index: c.index as u32,
                message: Message::Assistant {
                    content: Some(c.text),
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
        let aa_req = self.convert_request(request);
        let url = format!("{}/completions", self.base_url);

        let mut req = self
            .http_client
            .post(&url)
            .json(&aa_req)
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

        let aa_resp: AlephAlphaResponse = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        Ok(self.convert_response(aa_resp))
    }
}

#[async_trait::async_trait]
impl Provider for AlephAlphaClient {
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        self.make_request(request).await
    }

    async fn embeddings(&self, _request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        Err(LlmError::ProviderError(
            "Embeddings not implemented for Aleph Alpha".to_string(),
        ))
    }
    fn provider_name(&self) -> &'static str {
        "aleph_alpha"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aleph_alpha_client_creation() {
        let client = AlephAlphaClient::new("test-key");
        assert_eq!(client.provider_name(), "aleph_alpha");
    }

    #[test]
    fn test_request_conversion() {
        let client = AlephAlphaClient::new("test-key");

        let request = ChatCompletionRequest {
            model: "luminous-extended".to_string(),
            messages: vec![Message::User {
                content: "Hello!".to_string(),
                name: None,
            }],
            temperature: Some(0.5),
            max_tokens: Some(200),
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

        let aa_req = client.convert_request(request);

        assert_eq!(aa_req.model, "luminous-extended");
        assert!(aa_req.prompt.contains("Hello!"));
        assert_eq!(aa_req.temperature, Some(0.5));
    }
}

