//! Cohere API client for LLMG
//!
//! Implements the Provider trait for Cohere's API.

use llmg_core::{
    provider::{ApiKeyCredentials, Credentials, LlmError, Provider},
    types::{
        ChatCompletionRequest, ChatCompletionResponse, Choice, Embedding, EmbeddingRequest,
        EmbeddingResponse, Message, Usage,
    },
};
// use serde::{Serialize, Deserialize};

/// Cohere API client
#[derive(Debug)]
pub struct CohereClient {
    http_client: reqwest::Client,
    base_url: String,
    credentials: Box<dyn Credentials>,
}

/// Cohere chat request format
#[derive(Debug, serde::Serialize)]
struct CohereChatRequest {
    message: String,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    preamble: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    chat_history: Vec<CohereMessage>,
}

#[derive(Debug, serde::Serialize)]
struct CohereMessage {
    role: String,
    message: String,
}

/// Cohere chat response format
#[derive(Debug, serde::Deserialize)]
struct CohereChatResponse {
    text: String,
    #[serde(rename = "finish_reason")]
    finish_reason: String,
    meta: Option<CohereMeta>,
}

#[derive(Debug, serde::Deserialize)]
struct CohereMeta {
    #[serde(rename = "billed_units")]
    billed_units: Option<CohereBilledUnits>,
}

#[derive(Debug, serde::Deserialize)]
struct CohereBilledUnits {
    input_tokens: u32,
    output_tokens: u32,
}

/// Cohere embed request format
#[derive(Debug, serde::Serialize)]
struct CohereEmbedRequest {
    texts: Vec<String>,
    model: String,
    input_type: String,
}

/// Cohere embed response format
#[derive(Debug, serde::Deserialize)]
struct CohereEmbedResponse {
    id: String,
    embeddings: Vec<Vec<f32>>,
    #[serde(default)]
    meta: Option<CohereEmbedMeta>,
}

#[derive(Debug, serde::Deserialize)]
struct CohereEmbedMeta {
    billed_units: Option<CohereEmbedBilledUnits>,
}

#[derive(Debug, serde::Deserialize)]
struct CohereEmbedBilledUnits {
    input_tokens: u32,
}

impl CohereClient {
    /// Create a new Cohere client from environment
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("COHERE_API_KEY").map_err(|_| LlmError::AuthError)?;

        Ok(Self::new(api_key))
    }

    /// Create a new Cohere client with explicit API key
    pub fn new(api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();

        Self {
            http_client: reqwest::Client::new(),
            base_url: "https://api.cohere.ai/v1".to_string(),
            credentials: Box::new(ApiKeyCredentials::bearer(api_key)),
        }
    }

    /// Create with custom base URL
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Convert OpenAI format to Cohere format
    fn convert_request(&self, request: ChatCompletionRequest) -> CohereChatRequest {
        let mut chat_history = Vec::new();
        let mut preamble = None;
        let mut current_message = String::new();

        for msg in request.messages {
            match msg {
                Message::System { content, .. } => {
                    preamble = Some(content);
                }
                Message::User { content, .. } => {
                    if !current_message.is_empty() {
                        // Save previous assistant message if exists
                        chat_history.push(CohereMessage {
                            role: "CHATBOT".to_string(),
                            message: current_message.clone(),
                        });
                        current_message.clear();
                    }
                    current_message = content;
                }
                Message::Assistant {
                    content: Some(content),
                    ..
                } => {
                    if !current_message.is_empty() {
                        // Save user message
                        chat_history.push(CohereMessage {
                            role: "USER".to_string(),
                            message: current_message.clone(),
                        });
                        current_message.clear();
                    }
                    current_message = content;
                }
                _ => {}
            }
        }

        CohereChatRequest {
            message: current_message,
            model: request.model,
            preamble,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            p: request.top_p,
            stop_sequences: request.stop,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            stream: request.stream,
            chat_history,
        }
    }

    /// Convert Cohere response to OpenAI format
    fn convert_response(&self, response: CohereChatResponse) -> ChatCompletionResponse {
        let usage = response.meta.and_then(|m| m.billed_units).map(|u| Usage {
            prompt_tokens: u.input_tokens,
            completion_tokens: u.output_tokens,
            total_tokens: u.input_tokens + u.output_tokens,
        });

        ChatCompletionResponse {
            id: format!("cohere-{})", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: "cohere-command".to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message::Assistant {
                    content: Some(response.text),
                    refusal: None,
                    tool_calls: None,
                },
                finish_reason: Some(response.finish_reason),
            }],
            usage,
        }
    }

    async fn make_request(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        let cohere_req = self.convert_request(request);
        let url = format!("{}/chat", self.base_url);

        let mut req = self
            .http_client
            .post(&url)
            .json(&cohere_req)
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

        let cohere_resp: CohereChatResponse = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        Ok(self.convert_response(cohere_resp))
    }
}

#[async_trait::async_trait]
impl Provider for CohereClient {
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        self.make_request(request).await
    }

    async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        let cohere_req = CohereEmbedRequest {
            texts: vec![request.input],
            model: request.model.clone(),
            input_type: "search_document".to_string(),
        };
        let url = format!("{}/embed", self.base_url);

        let mut req = self
            .http_client
            .post(&url)
            .json(&cohere_req)
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

        let cohere_resp: CohereEmbedResponse = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        let input_tokens = cohere_resp
            .meta
            .and_then(|m| m.billed_units)
            .map(|u| u.input_tokens)
            .unwrap_or(0);

        Ok(EmbeddingResponse {
            id: cohere_resp.id,
            object: "list".to_string(),
            data: cohere_resp
                .embeddings
                .into_iter()
                .enumerate()
                .map(|(i, embedding)| Embedding {
                    index: i as u32,
                    object: "embedding".to_string(),
                    embedding,
                })
                .collect(),
            model: request.model,
            usage: Usage {
                prompt_tokens: input_tokens,
                completion_tokens: 0,
                total_tokens: input_tokens,
            },
        })
    }
    fn provider_name(&self) -> &'static str {
        "cohere"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cohere_client_creation() {
        let client = CohereClient::new("test-key");
        assert_eq!(client.provider_name(), "cohere");
    }

    #[test]
    fn test_request_conversion() {
        let client = CohereClient::new("test-key");

        let request = ChatCompletionRequest {
            model: "command".to_string(),
            messages: vec![
                Message::System {
                    content: "Be helpful".to_string(),
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

        let cohere_req = client.convert_request(request);

        assert_eq!(cohere_req.model, "command");
        assert_eq!(cohere_req.preamble, Some("Be helpful".to_string()));
        assert_eq!(cohere_req.message, "Hello!");
        assert_eq!(cohere_req.temperature, Some(0.7));
        assert_eq!(cohere_req.max_tokens, Some(100));
    }
}
