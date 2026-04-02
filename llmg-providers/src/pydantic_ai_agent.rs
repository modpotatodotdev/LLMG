use llmg_core::{
    provider::{ApiKeyCredentials, Credentials, LlmError, Provider},
    types::{
        ChatCompletionRequest, ChatCompletionResponse, Choice, EmbeddingRequest, EmbeddingResponse,
        Message, Usage,
    },
};
// use serde::{Deserialize, Serialize}; // removed unused imports

/// Pydantic AI Agent API client
#[derive(Debug)]
pub struct PydanticAiAgentClient {
    http_client: reqwest::Client,
    base_url: String,
    credentials: Box<dyn Credentials>,
}

/// Pydantic AI Agent-specific request format
#[derive(Debug, serde::Serialize)]
struct PydanticAiAgentRequest {
    model: String,
    messages: Vec<PydanticAiAgentMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

/// Pydantic AI Agent message format
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct PydanticAiAgentMessage {
    role: String,
    content: String,
}

/// Pydantic AI Agent response format
#[derive(Debug, serde::Deserialize)]
struct PydanticAiAgentResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<PydanticAiAgentChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<PydanticAiAgentUsage>,
}

#[derive(Debug, serde::Deserialize)]
struct PydanticAiAgentChoice {
    index: u32,
    message: PydanticAiAgentMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct PydanticAiAgentUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl PydanticAiAgentClient {
    /// Create a new Pydantic AI Agent client from environment
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("PYDANTIC_AI_API_KEY").map_err(|_| LlmError::AuthError)?;

        Ok(Self::new(api_key))
    }

    /// Create a new Pydantic AI Agent client with explicit API key
    pub fn new(api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();

        Self {
            http_client: reqwest::Client::new(),
            base_url: "https://api.pydantic-ai.com".to_string(),
            credentials: Box::new(ApiKeyCredentials::new(api_key)),
        }
    }

    /// Create with custom base URL
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Convert OpenAI format to Pydantic AI Agent format
    fn convert_request(&self, request: ChatCompletionRequest) -> PydanticAiAgentRequest {
        let mut system = None;
        let mut messages = Vec::new();

        for msg in request.messages {
            match msg {
                Message::System { content, .. } => {
                    system = Some(content);
                }
                Message::User { content, .. } => {
                    messages.push(PydanticAiAgentMessage {
                        role: "user".to_string(),
                        content,
                    });
                }
                Message::Assistant {
                    content: Some(content),
                    ..
                } => {
                    messages.push(PydanticAiAgentMessage {
                        role: "assistant".to_string(),
                        content,
                    });
                }
                _ => {}
            }
        }

        PydanticAiAgentRequest {
            model: request.model,
            messages,
            system,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            stream: request.stream,
        }
    }

    /// Convert Pydantic AI Agent response to OpenAI format
    fn convert_response(&self, response: PydanticAiAgentResponse) -> ChatCompletionResponse {
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
        let pydantic_req = self.convert_request(request);
        let url = format!("{}/v1/chat/completions", self.base_url);

        let mut req = self
            .http_client
            .post(&url)
            .json(&pydantic_req)
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

        let pydantic_resp: PydanticAiAgentResponse = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        Ok(self.convert_response(pydantic_resp))
    }
}

#[async_trait::async_trait]
impl Provider for PydanticAiAgentClient {
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        self.make_request(request).await
    }

    async fn embeddings(&self, _request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        Err(LlmError::ProviderError(
            "Pydantic AI Agent does not support embeddings".to_string(),
        ))
    }
    fn provider_name(&self) -> &'static str {
        "pydantic_ai_agent"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pydantic_ai_agent_client_creation() {
        let client = PydanticAiAgentClient::new("test-key");
        assert_eq!(client.provider_name(), "pydantic_ai_agent");
    }

    #[test]
    fn test_request_conversion() {
        let client = PydanticAiAgentClient::new("test-key");

        let request = ChatCompletionRequest {
            model: "pydantic-ai-agent-v1".to_string(),
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
            temperature: None,
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

        let pydantic_req = client.convert_request(request);

        assert_eq!(pydantic_req.model, "pydantic-ai-agent-v1");
        assert_eq!(
            pydantic_req.system,
            Some("You are a helpful assistant".to_string())
        );
        assert_eq!(pydantic_req.messages.len(), 1);
        assert_eq!(pydantic_req.messages[0].role, "user");
    }
}

