use llmg_core::{
    provider::{ApiKeyCredentials, Credentials, LlmError, Provider},
    types::{
        ChatCompletionRequest, ChatCompletionResponse, Embedding, EmbeddingRequest,
        EmbeddingResponse, Usage,
    },
};

/// Infinity AI API client (embeddings-only provider)
#[derive(Debug)]
pub struct InfinityClient {
    http_client: reqwest::Client,
    base_url: String,
    credentials: Box<dyn Credentials>,
}

/// Infinity embeddings request format
#[derive(Debug, serde::Serialize)]
struct InfinityEmbeddingRequest {
    model: String,
    input: Vec<String>,
}

/// Infinity embeddings response format
#[derive(Debug, serde::Deserialize)]
struct InfinityEmbeddingResponse {
    object: String,
    data: Vec<InfinityEmbeddingData>,
    model: String,
    usage: InfinityUsage,
}

#[derive(Debug, serde::Deserialize)]
struct InfinityEmbeddingData {
    embedding: Vec<f32>,
    index: u32,
}

#[derive(Debug, serde::Deserialize)]
struct InfinityUsage {
    total_tokens: u32,
}

impl InfinityClient {
    /// Create a new Infinity client from environment
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("INFINITY_API_KEY").map_err(|_| LlmError::AuthError)?;

        Ok(Self::new(api_key))
    }

    /// Create a new Infinity client with explicit API key
    pub fn new(api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();

        Self {
            http_client: reqwest::Client::new(),
            base_url: "https://api.lmnr.ai/v1".to_string(),
            credentials: Box::new(ApiKeyCredentials::with_header(api_key, "Authorization")),
        }
    }

    /// Create with custom base URL
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
}

#[async_trait::async_trait]
impl Provider for InfinityClient {
    async fn chat_completion(
        &self,
        _request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        Err(LlmError::ProviderError(
            "Infinity does not support chat completions - it is an embedding-only provider"
                .to_string(),
        ))
    }

    async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        let url = format!("{}/embeddings", self.base_url);

        let infinity_req = InfinityEmbeddingRequest {
            model: request.model,
            input: vec![request.input],
        };

        let mut req = self
            .http_client
            .post(&url)
            .json(&infinity_req)
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

        let infinity_resp: InfinityEmbeddingResponse = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        let embeddings = infinity_resp
            .data
            .into_iter()
            .map(|d| Embedding {
                index: d.index,
                object: "embedding".to_string(),
                embedding: d.embedding,
            })
            .collect();

        Ok(EmbeddingResponse {
            id: format!("infinity-{}", uuid::Uuid::new_v4()),
            object: infinity_resp.object,
            data: embeddings,
            model: infinity_resp.model,
            usage: Usage {
                prompt_tokens: infinity_resp.usage.total_tokens,
                completion_tokens: 0,
                total_tokens: infinity_resp.usage.total_tokens,
            },
        })
    }
    fn provider_name(&self) -> &'static str {
        "infinity"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infinity_client_creation() {
        let client = InfinityClient::new("test-key");
        assert_eq!(client.provider_name(), "infinity");
    }
}
