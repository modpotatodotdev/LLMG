use futures::{StreamExt, TryStreamExt};
use llmg_core::{
    provider::{ApiKeyCredentials, ChatCompletionStream, Credentials, LlmError, Provider},
    streaming::{ChatCompletionChunk, ChoiceDelta, DeltaContent},
    types::{ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse},
};
use std::future::Future;
use std::pin::Pin;

/// Z.AI API client (GLM)
#[derive(Debug)]
pub struct ZaiClient {
    http_client: reqwest::Client,
    base_url: String,
    credentials: Box<dyn Credentials>,
}

impl ZaiClient {
    /// Create a new Z.AI client with explicit API key and default general endpoint
    pub fn new(api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();
        let credentials = Box::new(ApiKeyCredentials::bearer(api_key));

        Self {
            http_client: reqwest::Client::new(),
            base_url: "https://api.z.ai/api/paas/v4".to_string(),
            credentials,
        }
    }

    /// Create a new Z.AI client for the coding plan
    pub fn coding(api_key: impl Into<String>) -> Self {
        Self::new(api_key).with_base_url("https://api.z.ai/api/coding/paas/v4")
    }

    /// Create with custom base URL
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
            .map_err(|e| LlmError::HttpError(e.to_string()))
            .then(move |bytes_result| {
                let chunk_id = chunk_id.clone();
                let model = model.clone();
                async move {
                    match bytes_result {
                        Ok(bytes) => {
                            let text = String::from_utf8_lossy(&bytes);
                            parse_zai_sse_line(&text, &chunk_id, &model)
                        }
                        Err(e) => Err(LlmError::HttpError(e.to_string())),
                    }
                }
            })
            .try_filter_map(|chunk| async move { Ok(chunk) });

        Ok(Box::pin(stream) as ChatCompletionStream)
    }
}

#[async_trait::async_trait]
impl Provider for ZaiClient {
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

    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        let url = format!("{}/models", self.base_url);

        let mut req = self
            .http_client
            .get(&url)
            .build()
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        self.credentials.apply(&mut req)?;

        let response = self
            .http_client
            .execute(req)
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            return Ok(vec![]);
        }

        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        let models = body["data"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m["id"].as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        Ok(models)
    }

    fn provider_name(&self) -> &'static str {
        "zai"
    }
}

fn parse_zai_sse_line(
    line: &str,
    chunk_id: &str,
    model: &str,
) -> Result<Option<ChatCompletionChunk>, LlmError> {
    let line = line.trim();
    if line.is_empty() || line == "data: [DONE]" {
        return Ok(None);
    }

    if let Some(json_str) = line.strip_prefix("data: ") {
        if json_str.trim().is_empty() {
            return Ok(None);
        }

        let parsed: serde_json::Value =
            serde_json::from_str(json_str).map_err(LlmError::SerializationError)?;

        let choices = parsed
            .get("choices")
            .and_then(|c| c.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|choice| {
                        let index =
                            choice.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as u32;
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

        return Ok(Some(ChatCompletionChunk {
            id: chunk_id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: model.to_string(),
            choices,
        }));
    }

    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zai_client_creation() {
        let client = ZaiClient::new("test-key");
        assert_eq!(client.provider_name(), "zai");
        assert!(client.base_url.contains("paas/v4"));
    }

    #[test]
    fn test_zai_coding_client() {
        let client = ZaiClient::coding("test-key");
        assert!(client.base_url.contains("coding/paas/v4"));
    }
}
