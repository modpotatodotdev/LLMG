use eventsource_stream::Eventsource;
use futures::{StreamExt, TryStreamExt};
use llmg_core::{
    provider::{ApiKeyCredentials, ChatCompletionStream, Credentials, LlmError, Provider},
    streaming::{ChatCompletionChunk, ChoiceDelta, DeltaContent},
    types::{ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse},
};
use std::future::Future;
use std::pin::Pin;

/// OpenAI API client
#[derive(Debug)]
pub struct OpenAiClient {
    http_client: reqwest::Client,
    base_url: String,
    credentials: Box<dyn Credentials>,
}

impl OpenAiClient {
    /// Create a new OpenAI client from environment
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| LlmError::AuthError)?;

        Ok(Self::new(api_key))
    }

    /// Create a new OpenAI client with explicit API key
    pub fn new(api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();
        let credentials = Box::new(ApiKeyCredentials::bearer(api_key));

        Self {
            http_client: reqwest::Client::new(),
            base_url: "https://api.openai.com/v1".to_string(),
            credentials,
        }
    }

    /// Create with custom base URL (for Azure or proxies)
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    async fn make_request(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        let url = format!("{}/chat/completions", self.base_url);

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
            .json::<ChatCompletionResponse>()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))
    }

    async fn make_stream_request(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionStream, LlmError> {
        let url = format!("{}/chat/completions", self.base_url);
        let mut stream_request = request;
        stream_request.stream = Some(true);

        let mut req = self
            .http_client
            .post(&url)
            .json(&stream_request)
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

        let chunk_id = ChatCompletionChunk::generate_id();
        let model = stream_request.model.clone();

        let stream = response
            .bytes_stream()
            .eventsource()
            .map_err(|e| LlmError::HttpError(e.to_string()))
            .then(move |event_result| {
                let chunk_id = chunk_id.clone();
                let model = model.clone();
                async move {
                    match event_result {
                        Ok(event) => parse_openai_sse_data(&event.data, &chunk_id, &model),
                        Err(e) => Err(LlmError::HttpError(e.to_string())),
                    }
                }
            })
            .try_filter_map(|chunk| async move { Ok(chunk) });

        Ok(Box::pin(stream) as ChatCompletionStream)
    }
}

#[async_trait::async_trait]
impl Provider for OpenAiClient {
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        self.make_request(request).await
    }

    fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ChatCompletionStream, LlmError>> + Send + '_>> {
        Box::pin(self.make_stream_request(request))
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
        "openai"
    }
}

fn parse_openai_sse_data(
    data: &str,
    chunk_id: &str,
    model: &str,
) -> Result<Option<ChatCompletionChunk>, LlmError> {
    let data = data.trim();
    if data.is_empty() || data == "[DONE]" {
        return Ok(None);
    }

    let parsed: serde_json::Value =
        serde_json::from_str(data).map_err(LlmError::SerializationError)?;

    let choices = parsed
        .get("choices")
        .and_then(|c| c.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|choice| {
                    let index = choice.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as u32;
                    let delta = choice.get("delta")?;
                    let finish_reason = choice
                        .get("finish_reason")
                        .and_then(|f| f.as_str())
                        .map(|s| s.to_string());

                    let role = delta
                        .get("role")
                        .and_then(|r| r.as_str())
                        .map(|s| s.to_string());
                    let content = delta
                        .get("content")
                        .and_then(|c| c.as_str())
                        .map(|s| s.to_string());

                    Some(ChoiceDelta {
                        index,
                        delta: DeltaContent { role, content },
                        finish_reason,
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    if choices.is_empty() {
        return Ok(None);
    }

    Ok(Some(ChatCompletionChunk {
        id: chunk_id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: model.to_string(),
        choices,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_client_creation() {
        let client = OpenAiClient::new("test-key");
        assert_eq!(client.provider_name(), "openai");
    }

    #[test]
    fn test_from_env_missing_key() {
        // Temporarily remove env var
        let original = std::env::var("OPENAI_API_KEY").ok();
        std::env::remove_var("OPENAI_API_KEY");

        let result = OpenAiClient::from_env();
        assert!(result.is_err());

        // Restore
        if let Some(key) = original {
            std::env::set_var("OPENAI_API_KEY", key);
        }
    }
}
