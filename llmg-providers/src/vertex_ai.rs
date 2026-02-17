use llmg_core::{
    provider::{ApiKeyCredentials, Credentials, LlmError, Provider},
    types::{
        ChatCompletionRequest, ChatCompletionResponse, Embedding, EmbeddingRequest,
        EmbeddingResponse, Usage,
    },
};
use serde::{Deserialize, Serialize};

/// Google Vertex AI API client
#[derive(Debug)]
pub struct VertexAiClient {
    http_client: reqwest::Client,
    base_url: String,
    credentials: Box<dyn Credentials>,
    project_id: String,
    location: String,
    access_token: Option<String>,
}

/// GCP OAuth token response
#[derive(Debug, Deserialize)]
struct GcpTokenResponse {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: u64,
    token_type: String,
}

/// Vertex AI request format
#[derive(Debug, Serialize)]
struct VertexAiRequest {
    contents: Vec<VertexAiContent>,
    generation_config: Option<VertexAiGenerationConfig>,
}

/// Vertex AI content format
#[derive(Debug, Serialize)]
struct VertexAiContent {
    role: String,
    parts: Vec<VertexAiPart>,
}

/// Vertex AI part format
#[derive(Debug, Serialize)]
struct VertexAiPart {
    text: String,
}

/// Vertex AI generation config
#[derive(Debug, Serialize)]
struct VertexAiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
}

/// Vertex AI embedding predict response
#[derive(Debug, Deserialize)]
struct VertexAiPredictResponse {
    predictions: Vec<VertexAiEmbeddingPrediction>,
}

#[derive(Debug, Deserialize)]
struct VertexAiEmbeddingPrediction {
    embeddings: VertexAiEmbeddingValues,
}

#[derive(Debug, Deserialize)]
struct VertexAiEmbeddingValues {
    values: Vec<f32>,
    statistics: Option<VertexAiEmbeddingStats>,
}

#[derive(Debug, Deserialize)]
struct VertexAiEmbeddingStats {
    token_count: Option<u32>,
}

/// Vertex AI response format
#[derive(Debug, Deserialize)]
struct VertexAiResponse {
    candidates: Vec<VertexAiCandidate>,
}

#[derive(Debug, Deserialize)]
struct VertexAiCandidate {
    content: VertexAiResponseContent,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct VertexAiResponseContent {
    parts: Vec<VertexAiResponsePart>,
}

#[derive(Debug, Deserialize)]
struct VertexAiResponsePart {
    text: String,
}

impl VertexAiClient {
    /// Create a new Google Vertex AI client from environment
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("GOOGLE_API_KEY").ok();
        let project_id = std::env::var("GOOGLE_CLOUD_PROJECT")
            .or_else(|_| std::env::var("GOOGLE_CLOUD_PROJECT_ID"))
            .map_err(|_| LlmError::InvalidRequest("GOOGLE_CLOUD_PROJECT not set".to_string()))?;
        let location =
            std::env::var("GOOGLE_CLOUD_LOCATION").unwrap_or_else(|_| "us-central1".to_string());
        let _service_account_key = std::env::var("GOOGLE_APPLICATION_CREDENTIALS").ok();

        Ok(Self::new(api_key, None, project_id, location))
    }

    /// Create a new Google Vertex AI client with explicit configuration
    pub fn new(
        api_key: Option<String>,
        #[allow(unused_variables)] _service_account_key: Option<String>,
        project_id: impl Into<String>,
        location: impl Into<String>,
    ) -> Self {
        let location_str = location.into();
        let project_id_str = project_id.into();
        Self {
            http_client: reqwest::Client::new(),
            base_url: format!(
                "https://{}.aiplatform.googleapis.com/v1/projects/{}",
                location_str, project_id_str
            ),
            credentials: Box::new(ApiKeyCredentials::with_header(
                api_key.unwrap_or_default(),
                "x-goog-api-key",
            )),
            project_id: project_id_str,
            location: location_str,
            access_token: None,
        }
    }

    /// Set a custom project ID
    pub fn with_project_id(mut self, project_id: impl Into<String>) -> Self {
        self.project_id = project_id.into();
        self
    }

    /// Set a custom location
    pub fn with_location(mut self, location: impl Into<String>) -> Self {
        self.location = location.into();
        self
    }

    /// Exchange service account credentials for OAuth tokens
    pub async fn exchange_code_for_tokens(
        &self,
        _client_id: &str,
        _client_secret: &str,
        _code: &str,
        _redirect_uri: &str,
    ) -> Result<String, LlmError> {
        Err(LlmError::ProviderError(
            "OAuth token exchange not yet implemented".to_string(),
        ))
    }

    /// Fetch project ID from GCP
    pub async fn fetch_project_id(&self) -> Result<String, LlmError> {
        Ok(self.project_id.clone())
    }

    /// Build inference URL
    fn build_url(&self, model: &str) -> String {
        format!(
            "{}/locations/{}/publishers/google/models/{}:generateContent",
            self.base_url, self.location, model
        )
    }

    /// Convert OpenAI format to Vertex AI format
    fn convert_request(&self, request: ChatCompletionRequest) -> VertexAiRequest {
        let contents = request
            .messages
            .iter()
            .filter_map(|msg| {
                let (role, text) = match msg {
                    llmg_core::types::Message::User { content, .. } => {
                        (Some("user"), content.as_str())
                    }
                    llmg_core::types::Message::Assistant { content, .. } => {
                        (Some("assistant"), content.as_deref().unwrap_or(""))
                    }
                    llmg_core::types::Message::System { content, .. } => {
                        (Some("user"), content.as_str())
                    }
                    _ => (None, ""),
                };

                role.map(|r| VertexAiContent {
                    role: r.to_string(),
                    parts: vec![VertexAiPart {
                        text: text.to_string(),
                    }],
                })
            })
            .collect();

        let _model_id = request
            .model
            .split('/')
            .next_back()
            .unwrap_or(&request.model);

        VertexAiRequest {
            contents,
            generation_config: Some(VertexAiGenerationConfig {
                max_output_tokens: request.max_tokens,
                temperature: request.temperature,
                top_p: request.top_p,
            }),
        }
    }

    /// Convert Vertex AI response to OpenAI format
    fn convert_response(
        &self,
        response: VertexAiResponse,
        model: String,
    ) -> ChatCompletionResponse {
        let candidate = response.candidates.first();

        let content = candidate
            .and_then(|c| c.content.parts.first())
            .map(|p| p.text.clone())
            .unwrap_or_default();

        let finish_reason = candidate
            .and_then(|c| c.finish_reason.as_ref())
            .map(|r| match r.as_str() {
                "STOP" => "stop",
                "MAX_TOKENS" => "length",
                "SAFETY" => "content_filter",
                "RECITATION" => "content_filter",
                "OTHER" => "other",
                _ => "stop",
            })
            .unwrap_or("stop");

        ChatCompletionResponse {
            id: uuid::Uuid::new_v4().to_string(),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            choices: vec![llmg_core::types::Choice {
                index: 0,
                message: llmg_core::types::Message::Assistant {
                    content: Some(content),
                    refusal: None,
                    tool_calls: None,
                },
                finish_reason: Some(finish_reason.to_string()),
            }],
            usage: None,
        }
    }

    async fn get_access_token(&self) -> Result<String, LlmError> {
        if let Some(ref token) = self.access_token {
            return Ok(token.clone());
        }

        Ok(String::new())
    }

    async fn make_request(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        let model = request.model.clone();
        let model_id = model.split('/').next_back().unwrap_or(&model);
        let vertex_req = self.convert_request(request);
        let url = self.build_url(model_id);

        let mut req = self
            .http_client
            .post(&url)
            .json(&vertex_req)
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

        let vertex_resp: VertexAiResponse = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        Ok(self.convert_response(vertex_resp, model_id.to_string()))
    }
}

#[async_trait::async_trait]
impl Provider for VertexAiClient {
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        self.make_request(request).await
    }

    async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        let model_id = request
            .model
            .split('/')
            .next_back()
            .unwrap_or(&request.model)
            .to_string();

        let url = format!(
            "{}/locations/{}/publishers/google/models/{}:predict",
            self.base_url, self.location, model_id
        );

        let body = serde_json::json!({
            "instances": [{"content": request.input}]
        });

        let mut req = self
            .http_client
            .post(&url)
            .json(&body)
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

        let predict_resp: VertexAiPredictResponse = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        let mut token_count = 0u32;
        let embeddings: Vec<Embedding> = predict_resp
            .predictions
            .into_iter()
            .enumerate()
            .map(|(i, pred)| {
                if let Some(stats) = &pred.embeddings.statistics {
                    token_count += stats.token_count.unwrap_or(0);
                }
                Embedding {
                    index: i as u32,
                    object: "embedding".to_string(),
                    embedding: pred.embeddings.values,
                }
            })
            .collect();

        Ok(EmbeddingResponse {
            id: format!("vertex-emb-{}", uuid::Uuid::new_v4()),
            object: "list".to_string(),
            data: embeddings,
            model: model_id,
            usage: Usage {
                prompt_tokens: token_count,
                completion_tokens: 0,
                total_tokens: token_count,
            },
        })
    }
    fn provider_name(&self) -> &'static str {
        "vertex_ai"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertex_ai_client_creation() {
        let client = VertexAiClient::new(
            Some("test-key".to_string()),
            None,
            "test-project",
            "us-central1",
        );
        assert_eq!(client.provider_name(), "vertex_ai");
    }

    #[test]
    fn test_url_building() {
        let client = VertexAiClient::new(
            Some("test-key".to_string()),
            None,
            "test-project",
            "us-central1",
        );
        let url = client.build_url("gemini-2.0-flash-exp");
        assert!(url.contains("aiplatform.googleapis.com"));
        assert!(url.contains("test-project"));
        assert!(url.contains("us-central1"));
        assert!(url.contains("gemini-2.0-flash-exp"));
    }
}
