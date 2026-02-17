use llmg_core::{
    provider::{ApiKeyCredentials, Credentials, LlmError, Provider},
    types::{
        ChatCompletionRequest, ChatCompletionResponse, Embedding, EmbeddingRequest,
        EmbeddingResponse, Usage,
    },
};
use serde::{Deserialize, Serialize};

/// IBM WatsonX API client
#[derive(Debug)]
pub struct WatsonxClient {
    http_client: reqwest::Client,
    base_url: String,
    credentials: Box<dyn Credentials>,
    project_id: String,
    api_key: String,
    access_token: Option<String>,
}

/// IBM IAM token response
#[derive(Debug, Deserialize)]
struct IamTokenResponse {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: u64,
    token_type: String,
}

/// WatsonX request format
#[derive(Debug, Serialize)]
struct WatsonxRequest {
    model_id: String,
    inputs: Vec<String>,
    parameters: Option<WatsonxParameters>,
}

/// WatsonX inference parameters
#[derive(Debug, Serialize)]
struct WatsonxParameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_new_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
}

/// WatsonX response format
#[derive(Debug, Deserialize)]
struct WatsonxResponse {
    results: Vec<WatsonxResult>,
    model_id: String,
}

#[derive(Debug, Deserialize)]
struct WatsonxResult {
    generated_text: String,
    generated_token_count: Option<u32>,
    input_token_count: Option<u32>,
    input_text: String,
}

/// WatsonX embedding request format
#[derive(Debug, Serialize)]
struct WatsonxEmbeddingRequest {
    model_id: String,
    inputs: Vec<String>,
    project_id: String,
}

/// WatsonX embedding response format
#[derive(Debug, Deserialize)]
struct WatsonxEmbeddingResponse {
    results: Vec<WatsonxEmbeddingResult>,
}

#[derive(Debug, Deserialize)]
struct WatsonxEmbeddingResult {
    embedding: Vec<f32>,
    input_token_count: Option<u32>,
}

impl WatsonxClient {
    /// Create a new IBM WatsonX client from environment
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("WATSONX_API_KEY")
            .or_else(|_| std::env::var("IBM_CLOUD_API_KEY"))
            .map_err(|_| LlmError::AuthError)?;
        let project_id = std::env::var("WATSONX_PROJECT_ID")
            .map_err(|_| LlmError::InvalidRequest("WATSONX_PROJECT_ID not set".to_string()))?;
        let base_url = std::env::var("WATSONX_URL")
            .unwrap_or_else(|_| "https://us-south.cloud.ibm.com/watsonx".to_string());

        Ok(Self::new(api_key, base_url, project_id))
    }

    /// Create a new IBM WatsonX client with explicit configuration
    pub fn new(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        project_id: impl Into<String>,
    ) -> Self {
        Self {
            http_client: reqwest::Client::new(),
            base_url: base_url.into(),
            credentials: Box::new(ApiKeyCredentials::with_header(
                String::new(),
                "Authorization",
            )),
            project_id: project_id.into(),
            api_key: api_key.into(),
            access_token: None,
        }
    }

    /// Set a custom project ID
    pub fn with_project_id(mut self, project_id: impl Into<String>) -> Self {
        self.project_id = project_id.into();
        self
    }

    /// Exchange API key for IBM IAM access token
    pub async fn exchange_code_for_tokens(&self) -> Result<String, LlmError> {
        // Placeholder for IBM IAM token exchange
        // In production, this would:
        // 1. POST to https://iam.cloud.ibm.com/identity/token
        // 2. Include API key in grant_type=urn:ibm:params:oauth:grant-type:apikey
        // 3. Return access token

        let url = "https://iam.cloud.ibm.com/identity/token";

        let mut form = std::collections::HashMap::new();
        form.insert("grant_type", "urn:ibm:params:oauth:grant-type:apikey");
        form.insert("apikey", &self.api_key);

        let response = self
            .http_client
            .post(url)
            .form(&form)
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

        let token_response: IamTokenResponse = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        Ok(token_response.access_token)
    }

    /// Fetch project ID from WatsonX
    pub async fn fetch_project_id(&self) -> Result<String, LlmError> {
        // Placeholder for project ID fetching
        // In production, this would:
        // 1. Call WatsonX management API
        // 2. List available projects
        // 3. Return project ID

        Ok(self.project_id.clone())
    }

    /// Build inference URL
    fn build_url(&self, _model: &str) -> String {
        format!(
            "{}/ml/v1-beta/generation/text?version=2024-05-31&project_id={}",
            self.base_url, self.project_id
        )
    }

    /// Convert OpenAI format to WatsonX format
    fn convert_request(&self, request: ChatCompletionRequest) -> WatsonxRequest {
        let input = request
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

        let model_id = request
            .model
            .split('/')
            .next_back()
            .unwrap_or(&request.model);

        WatsonxRequest {
            model_id: model_id.to_string(),
            inputs: vec![input],
            parameters: Some(WatsonxParameters {
                max_new_tokens: request.max_tokens,
                temperature: request.temperature,
                top_p: request.top_p,
                top_k: None,
            }),
        }
    }

    /// Convert WatsonX response to OpenAI format
    fn convert_response(&self, response: WatsonxResponse, model: String) -> ChatCompletionResponse {
        let result = response.results.first();

        let generated_text = result.map(|r| r.generated_text.clone()).unwrap_or_default();

        ChatCompletionResponse {
            id: uuid::Uuid::new_v4().to_string(),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            choices: vec![llmg_core::types::Choice {
                index: 0,
                message: llmg_core::types::Message::Assistant {
                    content: Some(generated_text),
                    refusal: None,
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: result.map(|r| llmg_core::types::Usage {
                prompt_tokens: r.input_token_count.unwrap_or(0),
                completion_tokens: r.generated_token_count.unwrap_or(0),
                total_tokens: r
                    .input_token_count
                    .unwrap_or(0)
                    .saturating_add(r.generated_token_count.unwrap_or(0)),
            }),
        }
    }

    async fn get_access_token(&self) -> Result<String, LlmError> {
        if let Some(ref token) = self.access_token {
            return Ok(token.clone());
        }

        let token = self.exchange_code_for_tokens().await?;
        Ok(token)
    }

    async fn make_request(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        let watsonx_req = self.convert_request(request);
        let url = self.build_url(&watsonx_req.model_id);

        let mut req = self
            .http_client
            .post(&url)
            .json(&watsonx_req)
            .build()
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        let token = self.get_access_token().await?;
        req.headers_mut().insert(
            "Authorization",
            format!("Bearer {}", token).parse().unwrap(),
        );

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

        let watsonx_resp: WatsonxResponse = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        Ok(self.convert_response(watsonx_resp, watsonx_req.model_id))
    }
}

#[async_trait::async_trait]
impl Provider for WatsonxClient {
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

        let url = format!("{}/ml/v1/text/embeddings?version=2024-03-14", self.base_url);

        let embed_req = WatsonxEmbeddingRequest {
            model_id: model_id.clone(),
            inputs: vec![request.input],
            project_id: self.project_id.clone(),
        };

        let token = self.get_access_token().await?;

        let mut req = self
            .http_client
            .post(&url)
            .json(&embed_req)
            .build()
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        req.headers_mut().insert(
            "Authorization",
            format!("Bearer {}", token).parse().unwrap(),
        );

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

        let embed_resp: WatsonxEmbeddingResponse = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        let mut total_tokens = 0u32;
        let embeddings: Vec<Embedding> = embed_resp
            .results
            .into_iter()
            .enumerate()
            .map(|(i, result)| {
                total_tokens += result.input_token_count.unwrap_or(0);
                Embedding {
                    index: i as u32,
                    object: "embedding".to_string(),
                    embedding: result.embedding,
                }
            })
            .collect();

        Ok(EmbeddingResponse {
            id: format!("watsonx-emb-{}", uuid::Uuid::new_v4()),
            object: "list".to_string(),
            data: embeddings,
            model: model_id,
            usage: Usage {
                prompt_tokens: total_tokens,
                completion_tokens: 0,
                total_tokens,
            },
        })
    }
    fn provider_name(&self) -> &'static str {
        "watsonx"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_watsonx_client_creation() {
        let client = WatsonxClient::new(
            "test-key",
            "https://us-south.cloud.ibm.com/watsonx",
            "test-project",
        );
        assert_eq!(client.provider_name(), "watsonx");
    }

    #[test]
    fn test_url_building() {
        let client = WatsonxClient::new(
            "test-key",
            "https://us-south.cloud.ibm.com/watsonx",
            "my-project",
        );
        let url = client.build_url("test-model");
        assert!(url.contains("us-south.cloud.ibm.com/watsonx"));
        assert!(url.contains("ml/v1-beta/generation/text"));
        assert!(url.contains("project_id=my-project"));
    }
}
