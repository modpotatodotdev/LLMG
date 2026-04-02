use llmg_core::{
    provider::{ApiKeyCredentials, Credentials, LlmError, Provider},
    types::{ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse},
};
use serde::{Deserialize, Serialize};

/// Heroku AI API client
#[derive(Debug)]
pub struct HerokuClient {
    http_client: reqwest::Client,
    base_url: String,
    credentials: Box<dyn Credentials>,
    app_name: Option<String>,
}

/// Heroku request format
#[derive(Debug, Serialize)]
struct HerokuRequest {
    model: String,
    messages: Vec<HerokuMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

/// Heroku message format
#[derive(Debug, Serialize)]
struct HerokuMessage {
    role: String,
    content: String,
}

/// Heroku response format
#[derive(Debug, Deserialize)]
struct HerokuResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<HerokuChoice>,
    usage: HerokuUsage,
}

#[derive(Debug, Deserialize)]
struct HerokuChoice {
    index: u32,
    message: HerokuResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct HerokuResponseMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct HerokuUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl HerokuClient {
    /// Create a new Heroku client from environment
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("HEROKU_API_KEY").map_err(|_| LlmError::AuthError)?;
        let app_name = std::env::var("HEROKU_APP_NAME").ok();
        let base_url = std::env::var("HEROKU_BASE_URL")
            .unwrap_or_else(|_| "https://api.heroku.com".to_string());

        Ok(Self::new(api_key, base_url, app_name))
    }

    /// Create a new Heroku client with explicit configuration
    pub fn new(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        app_name: Option<String>,
    ) -> Self {
        Self {
            http_client: reqwest::Client::new(),
            base_url: base_url.into(),
            credentials: Box::new(ApiKeyCredentials::with_header(api_key, "Authorization")),
            app_name,
        }
    }

    /// Set a custom app name
    pub fn with_app_name(mut self, app_name: impl Into<String>) -> Self {
        self.app_name = Some(app_name.into());
        self
    }

    /// Build chat completions URL
    fn build_url(&self) -> String {
        if let Some(ref app) = self.app_name {
            format!("{}/apps/{}/ai/completions", self.base_url, app)
        } else {
            format!("{}/v1/completions", self.base_url)
        }
    }

    /// Convert OpenAI format to Heroku format
    fn convert_request(&self, request: ChatCompletionRequest) -> HerokuRequest {
        let messages = request
            .messages
            .into_iter()
            .map(|msg| match msg {
                llmg_core::types::Message::User { content, .. } => HerokuMessage {
                    role: "user".to_string(),
                    content,
                },
                llmg_core::types::Message::Assistant { content, .. } => HerokuMessage {
                    role: "assistant".to_string(),
                    content: content.unwrap_or_default(),
                },
                llmg_core::types::Message::System { content, .. } => HerokuMessage {
                    role: "system".to_string(),
                    content,
                },
                _ => HerokuMessage {
                    role: "user".to_string(),
                    content: String::new(),
                },
            })
            .collect();

        HerokuRequest {
            model: request.model,
            messages,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            stream: request.stream,
        }
    }

    /// Convert Heroku response to OpenAI format
    fn convert_response(&self, response: HerokuResponse) -> ChatCompletionResponse {
        ChatCompletionResponse {
            id: response.id,
            object: response.object,
            created: response.created as i64,
            model: response.model,
            choices: response
                .choices
                .into_iter()
                .map(|c| llmg_core::types::Choice {
                    index: c.index,
                    message: llmg_core::types::Message::Assistant {
                        content: Some(c.message.content),
                        refusal: None,
                        tool_calls: None,
                    },
                    finish_reason: c.finish_reason,
                })
                .collect(),
            usage: Some(llmg_core::types::Usage {
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
        let heroku_req = self.convert_request(request);
        let url = self.build_url();

        let mut req = self
            .http_client
            .post(&url)
            .json(&heroku_req)
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

        let heroku_resp: HerokuResponse = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        Ok(self.convert_response(heroku_resp))
    }
}

#[async_trait::async_trait]
impl Provider for HerokuClient {
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
        "heroku"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heroku_client_creation() {
        let client = HerokuClient::new(
            "test-key",
            "https://api.heroku.com",
            Some("my-app".to_string()),
        );
        assert_eq!(client.provider_name(), "heroku");
    }

    #[test]
    fn test_url_building() {
        let client = HerokuClient::new(
            "test-key",
            "https://api.heroku.com",
            Some("my-app".to_string()),
        );
        let url = client.build_url();
        assert!(url.contains("api.heroku.com"));
        assert!(url.contains("my-app"));
        assert!(url.contains("completions"));
    }
}

