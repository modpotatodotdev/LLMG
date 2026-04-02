use llmg_core::{
    provider::{ApiKeyCredentials, Credentials, LlmError, Provider},
    types::{
        ChatCompletionRequest, ChatCompletionResponse, Embedding, EmbeddingRequest,
        EmbeddingResponse, Usage,
    },
};

/// Voyage AI API client
#[derive(Debug)]
pub struct VoyageaiClient {
    http_client: reqwest::Client,
    base_url: String,
    credentials: Box<dyn Credentials>,
}

/// Voyage AI embeddings request format
#[derive(Debug, serde::Serialize)]
struct VoyageaiEmbeddingRequest {
    model: String,
    input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_type: Option<String>,
}

/// Voyage AI embeddings response format
#[derive(Debug, serde::Deserialize)]
struct VoyageaiEmbeddingResponse {
    object: String,
    data: Vec<VoyageaiEmbeddingData>,
    model: String,
    usage: VoyageaiUsage,
}

#[derive(Debug, serde::Deserialize)]
struct VoyageaiEmbeddingData {
    embedding: Vec<f32>,
    index: u32,
}

#[derive(Debug, serde::Deserialize)]
struct VoyageaiUsage {
    total_tokens: u32,
}

impl VoyageaiClient {
    /// Create a new Voyage AI client from environment
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("VOYAGE_API_KEY").map_err(|_| LlmError::AuthError)?;

        Ok(Self::new(api_key))
    }

    /// Create a new Voyage AI client with explicit API key
    pub fn new(api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();

        Self {
            http_client: reqwest::Client::new(),
            base_url: "https://api.voyageai.com/v1".to_string(),
            credentials: Box::new(ApiKeyCredentials::new(api_key)),
        }
    }

    /// Create with custom base URL
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
}

#[async_trait::async_trait]
impl Provider for VoyageaiClient {
    async fn chat_completion(
        &self,
        _request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        Err(LlmError::ProviderError(
            "Voyage AI does not support chat completions - it is an embedding-only provider"
                .to_string(),
        ))
    }

    async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        let url = format!("{}/embeddings", self.base_url);

        let voyage_req = VoyageaiEmbeddingRequest {
            model: request.model,
            input: vec![request.input],
            input_type: Some("document".to_string()),
        };

        let mut req = self
            .http_client
            .post(&url)
            .json(&voyage_req)
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

        let voyage_resp: VoyageaiEmbeddingResponse = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        let embeddings = voyage_resp
            .data
            .into_iter()
            .map(|d| Embedding {
                index: d.index,
                object: "embedding".to_string(),
                embedding: d.embedding,
            })
            .collect();

        Ok(EmbeddingResponse {
            id: format!("voyage-{}", uuid::Uuid::new_v4()),
            object: voyage_resp.object,
            data: embeddings,
            model: voyage_resp.model,
            usage: Usage {
                prompt_tokens: voyage_resp.usage.total_tokens,
                completion_tokens: 0,
                total_tokens: voyage_resp.usage.total_tokens,
            },
        })
    }
    fn provider_name(&self) -> &'static str {
        "voyageai"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voyageai_client_creation() {
        let client = VoyageaiClient::new("test-key");
        assert_eq!(client.provider_name(), "voyageai");
    }
}

