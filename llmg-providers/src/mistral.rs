use llmg_core::{
    provider::{ApiKeyCredentials, Credentials, LlmError, Provider},
    types::{
        ChatCompletionRequest, ChatCompletionResponse, Choice, EmbeddingRequest, EmbeddingResponse,
        Message, Usage,
    },
};

/// Mistral AI API client
#[derive(Debug)]
pub struct MistralClient {
    http_client: reqwest::Client,
    base_url: String,
    credentials: Box<dyn Credentials>,
}

/// Mistral-specific request format
#[derive(Debug, serde::Serialize)]
struct MistralRequest {
    model: String,
    messages: Vec<MistralMessage>,
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

/// Mistral message format
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct MistralMessage {
    role: String,
    content: String,
}

/// Mistral response format
#[derive(Debug, serde::Deserialize)]
struct MistralResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<MistralChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<MistralUsage>,
}

#[derive(Debug, serde::Deserialize)]
struct MistralChoice {
    index: u32,
    message: MistralMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct MistralUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl MistralClient {
    /// Create a new Mistral client from environment
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("MISTRAL_API_KEY").map_err(|_| LlmError::AuthError)?;

        Ok(Self::new(api_key))
    }

    /// Create a new Mistral client with explicit API key
    pub fn new(api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();

        Self {
            http_client: reqwest::Client::new(),
            base_url: "https://api.mistral.ai/v1".to_string(),
            credentials: Box::new(ApiKeyCredentials::new(api_key)),
        }
    }

    /// Create with custom base URL
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Convert OpenAI format to Mistral format
    fn convert_request(&self, request: ChatCompletionRequest) -> MistralRequest {
        let messages = request
            .messages
            .into_iter()
            .filter_map(|msg| match msg {
                Message::System { content, .. } => Some(MistralMessage {
                    role: "system".to_string(),
                    content,
                }),
                Message::User { content, .. } => Some(MistralMessage {
                    role: "user".to_string(),
                    content,
                }),
                Message::Assistant { content, .. } => content.map(|c| MistralMessage {
                    role: "assistant".to_string(),
                    content: c,
                }),
                _ => None,
            })
            .collect();

        MistralRequest {
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

    /// Convert Mistral response to OpenAI format
    fn convert_response(&self, response: MistralResponse) -> ChatCompletionResponse {
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
        let mistral_req = self.convert_request(request);
        let url = format!("{}/chat/completions", self.base_url);

        let mut req = self
            .http_client
            .post(&url)
            .json(&mistral_req)
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

        let mistral_resp: MistralResponse = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        Ok(self.convert_response(mistral_resp))
    }
}

#[async_trait::async_trait]
impl Provider for MistralClient {
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
        "mistral"
    }

    fn supported_models(&self) -> Vec<String> {
        vec![
            "mistral-large-latest".to_string(),
            "mistral-large-2407".to_string(),
            "mistral-large-2411".to_string(),
            "mistral-medium-latest".to_string(),
            "mistral-medium-2312".to_string(),
            "mistral-small-latest".to_string(),
            "mistral-small-2402".to_string(),
            "mistral-small-2409".to_string(),
            "open-mistral-nemo".to_string(),
            "open-mistral-7b".to_string(),
            "open-mixtral-8x7b".to_string(),
            "open-mixtral-8x22b".to_string(),
            "codestral-latest".to_string(),
            "codestral-2405".to_string(),
            "pixtral-large-latest".to_string(),
            "pixtral-12b".to_string(),
            "ministral-3b-latest".to_string(),
            "ministral-8b-latest".to_string(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mistral_client_creation() {
        let client = MistralClient::new("test-key");
        assert_eq!(client.provider_name(), "mistral");
    }

    #[test]
    fn test_request_conversion() {
        let client = MistralClient::new("test-key");

        let request = ChatCompletionRequest {
            model: "mistral-large-3".to_string(),
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

        let mistral_req = client.convert_request(request);

        assert_eq!(mistral_req.model, "mistral-large-3");
        assert_eq!(mistral_req.messages.len(), 1);
        assert_eq!(mistral_req.messages[0].role, "user");
    }
}
