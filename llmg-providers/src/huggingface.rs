use llmg_core::{
    provider::{ApiKeyCredentials, Credentials, LlmError, Provider},
    types::{ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse},
};
use serde::{Deserialize, Serialize};

/// HuggingFace Inference API client
#[derive(Debug)]
pub struct HuggingFaceClient {
    http_client: reqwest::Client,
    base_url: String,
    credentials: Box<dyn Credentials>,
    hf_token: Option<String>,
    api_key: Option<String>,
}

/// HuggingFace request format
#[derive(Debug, Serialize)]
struct HuggingFaceRequest {
    inputs: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<HuggingFaceParameters>,
}

/// HuggingFace inference parameters
#[derive(Debug, Serialize)]
struct HuggingFaceParameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_new_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    return_full_text: Option<bool>,
}

/// HuggingFace response format
#[derive(Debug, Deserialize)]
struct HuggingFaceResponse {
    #[serde(default)]
    generated_text: String,
}

/// HuggingFace streaming chunk
#[derive(Debug, Deserialize)]
struct HuggingFaceStreamChunk {
    token: Option<serde_json::Value>,
}

/// HuggingFace embeddings response format (flat array of vectors)
type HuggingFaceEmbeddingResponse = Vec<Vec<f32>>;

impl HuggingFaceClient {
    /// Create a new HuggingFace client from environment
    pub fn from_env() -> Result<Self, LlmError> {
        let hf_token = std::env::var("HF_TOKEN").ok();
        let api_key = std::env::var("HF_API_KEY")
            .or_else(|_| std::env::var("HUGGINGFACE_API_KEY"))
            .ok();

        if hf_token.is_none() && api_key.is_none() {
            return Err(LlmError::AuthError);
        }

        let base_url = std::env::var("HF_BASE_URL")
            .unwrap_or_else(|_| "https://api-inference.huggingface.co".to_string());

        Ok(Self::new(base_url, hf_token, api_key))
    }

    /// Create a new HuggingFace client with explicit configuration
    pub fn new(
        base_url: impl Into<String>,
        hf_token: Option<String>,
        api_key: Option<String>,
    ) -> Self {
        let credentials = if let Some(token) = hf_token.clone() {
            ApiKeyCredentials::with_header(token, "Authorization")
        } else if let Some(key) = api_key.clone() {
            ApiKeyCredentials::with_header(key, "Authorization")
        } else {
            ApiKeyCredentials::new(String::new())
        };

        Self {
            http_client: reqwest::Client::new(),
            base_url: base_url.into(),
            credentials: Box::new(credentials),
            hf_token,
            api_key,
        }
    }

    /// Set a custom HF Token
    pub fn with_hf_token(mut self, token: impl Into<String>) -> Self {
        self.hf_token = Some(token.into());
        self.credentials = Box::new(ApiKeyCredentials::with_header(
            self.hf_token.as_ref().unwrap().clone(),
            "Authorization",
        ));
        self
    }

    /// Set a custom API Key
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self.credentials = Box::new(ApiKeyCredentials::with_header(
            self.api_key.as_ref().unwrap().clone(),
            "Authorization",
        ));
        self
    }

    /// Build model inference URL
    fn build_url(&self, model: &str) -> String {
        let model_id = model.split('/').next_back().unwrap_or(model);
        format!("{}/models/{}", self.base_url, model_id)
    }

    /// Convert OpenAI format to HuggingFace format
    fn convert_request(&self, request: ChatCompletionRequest) -> HuggingFaceRequest {
        let content = request
            .messages
            .iter()
            .map(|msg| match msg {
                llmg_core::types::Message::User { content, .. } => content.clone(),
                llmg_core::types::Message::Assistant { content, .. } => {
                    content.as_deref().unwrap_or("").to_string()
                }
                llmg_core::types::Message::System { content, .. } => {
                    format!("System: {}", content)
                }
                _ => String::new(),
            })
            .collect::<Vec<_>>()
            .join("\n");

        HuggingFaceRequest {
            inputs: content,
            parameters: Some(HuggingFaceParameters {
                max_new_tokens: request.max_tokens,
                temperature: request.temperature,
                top_p: request.top_p,
                return_full_text: Some(true),
            }),
        }
    }

    /// Convert HuggingFace response to OpenAI format
    fn convert_response(
        &self,
        response: HuggingFaceResponse,
        model: String,
    ) -> ChatCompletionResponse {
        ChatCompletionResponse {
            id: uuid::Uuid::new_v4().to_string(),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            choices: vec![llmg_core::types::Choice {
                index: 0,
                message: llmg_core::types::Message::Assistant {
                    content: Some(response.generated_text),
                    refusal: None,
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: None,
        }
    }

    /// Validate HF token
    pub fn validate_hf_token(&self) -> Result<(), LlmError> {
        if let Some(ref token) = self.hf_token {
            if token.is_empty() {
                return Err(LlmError::AuthError);
            }
        } else if let Some(ref key) = self.api_key {
            if key.is_empty() {
                return Err(LlmError::AuthError);
            }
        } else {
            return Err(LlmError::AuthError);
        }

        Ok(())
    }

    async fn make_request(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        self.validate_hf_token()?;

        let model_id = request
            .model
            .split('/')
            .next_back()
            .unwrap_or(&request.model);
        let hf_req = self.convert_request(request.clone());
        let url = self.build_url(model_id);

        let mut req = self
            .http_client
            .post(&url)
            .json(&hf_req)
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

        let hf_resp: HuggingFaceResponse = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        Ok(self.convert_response(hf_resp, model_id.to_string()))
    }
}

#[async_trait::async_trait]
impl Provider for HuggingFaceClient {
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        self.make_request(request).await
    }

    async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        let url = format!("{}/models/{}", self.base_url, request.model);

        let body = serde_json::json!({
            "inputs": request.input
        });

        let mut req_builder = self.http_client.post(&url).json(&body);

        if let Some(ref token) = self.hf_token {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", token));
        } else if let Some(ref key) = self.api_key {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", key));
        }

        let response = req_builder
            .send()
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

        let embeddings: HuggingFaceEmbeddingResponse = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        Ok(EmbeddingResponse {
            id: format!("hf-{}", uuid::Uuid::new_v4()),
            object: "list".to_string(),
            data: embeddings
                .into_iter()
                .enumerate()
                .map(|(i, embedding)| llmg_core::types::Embedding {
                    index: i as u32,
                    object: "embedding".to_string(),
                    embedding,
                })
                .collect(),
            model: request.model,
            usage: llmg_core::types::Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
        })
    }
    fn provider_name(&self) -> &'static str {
        "huggingface"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huggingface_client_creation() {
        let client = HuggingFaceClient::new(
            "https://api-inference.huggingface.co",
            Some("hf-test-token".to_string()),
            None,
        );
        assert_eq!(client.provider_name(), "huggingface");
    }

    #[test]
    fn test_hf_token_validation() {
        let client = HuggingFaceClient::new(
            "https://api-inference.huggingface.co",
            Some("hf-test-token".to_string()),
            None,
        );
        assert!(client.validate_hf_token().is_ok());
    }

    #[test]
    fn test_url_building() {
        let client = HuggingFaceClient::new(
            "https://api-inference.huggingface.co",
            Some("hf-test-token".to_string()),
            None,
        );
        let url = client.build_url("meta-llama/Llama-2-7b-chat-hf");
        assert!(url.contains("api-inference.huggingface.co"));
        assert!(url.contains("models"));
        assert!(url.contains("Llama-2-7b-chat-hf"));
    }
}

