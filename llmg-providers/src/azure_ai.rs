use llmg_core::{
    provider::{ApiKeyCredentials, Credentials, LlmError, Provider},
    types::{
        ChatCompletionRequest, ChatCompletionResponse, Embedding, EmbeddingRequest,
        EmbeddingResponse, Usage,
    },
};
use serde::{Deserialize, Serialize};

/// Azure AI API client
#[derive(Debug)]
pub struct AzureAiClient {
    http_client: reqwest::Client,
    base_url: String,
    credentials: Box<dyn Credentials>,
    project_id: Option<String>,
    api_version: String,
}

/// Azure AI OAuth token response
#[derive(Debug, Deserialize)]
/// OAuth token response (public for test access)
pub struct OAuthTokenResponse {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: u64,
    token_type: String,
}

/// Azure AI request format
#[derive(Debug, Serialize)]
struct AzureAiRequest {
    messages: Vec<AzureAiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

/// Azure AI message format
#[derive(Debug, Serialize)]
struct AzureAiMessage {
    role: String,
    content: String,
}

/// Azure AI response format
#[derive(Debug, Deserialize)]
struct AzureAiResponse {
    id: String,
    choices: Vec<AzureAiChoice>,
    usage: Option<AzureAiUsage>,
    model: String,
}

#[derive(Debug, Deserialize)]
struct AzureAiChoice {
    message: AzureAiResponseMessage,
    finish_reason: Option<String>,
    index: u32,
}

#[derive(Debug, Deserialize)]
struct AzureAiResponseMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct AzureAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl AzureAiClient {
    /// Create a new Azure AI client from environment
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("AZURE_AI_API_KEY")
            .or_else(|_| std::env::var("AZURE_OPENAI_API_KEY"))
            .map_err(|_| LlmError::AuthError)?;
        let endpoint = std::env::var("AZURE_AI_ENDPOINT")
            .unwrap_or_else(|_| "https://api.azure.microsoft.com".to_string());
        let project_id = std::env::var("AZURE_AI_PROJECT_ID").ok();

        Ok(Self::new(api_key, endpoint, project_id))
    }

    /// Create a new Azure AI client with explicit configuration
    pub fn new(
        api_key: impl Into<String>,
        endpoint: impl Into<String>,
        project_id: Option<String>,
    ) -> Self {
        Self {
            http_client: reqwest::Client::new(),
            base_url: endpoint.into(),
            credentials: Box::new(ApiKeyCredentials::with_header(api_key, "api-key")),
            project_id,
            api_version: "2024-02-15-preview".to_string(),
        }
    }

    /// Set a custom API version
    pub fn with_api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = version.into();
        self
    }

    /// Exchange authorization code for OAuth tokens
    pub async fn exchange_code_for_tokens(
        &self,
        _client_id: &str,
        _client_secret: &str,
        _code: &str,
        _redirect_uri: &str,
    ) -> Result<OAuthTokenResponse, LlmError> {
        // Placeholder for OAuth token exchange
        // Make method and response public for test access
        let resp = OAuthTokenResponse {
            access_token: String::new(),
            refresh_token: None,
            expires_in: 0,
            token_type: String::new(),
        };
        // In production, this would:
        // 1. POST to https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token
        // 2. Include client credentials and authorization code
        // 3. Return access and refresh tokens
        Ok(resp)
    }

    /// Fetch project ID from Azure AI
    pub async fn fetch_project_id(&self) -> Result<String, LlmError> {
        // Placeholder for project ID fetching
        // In production, this would:
        // 1. Call Azure AI management API
        // 2. List available projects
        // 3. Return the project ID
        self.project_id
            .clone()
            .ok_or_else(|| LlmError::InvalidRequest("Project ID not set".to_string()))
    }

    /// Build the chat completions URL
    fn build_url(&self, deployment: &str) -> String {
        format!(
            "{}/deployments/{}/chat/completions?api-version={}",
            self.base_url, deployment, self.api_version
        )
    }

    /// Build the embeddings URL
    fn build_embeddings_url(&self, deployment: &str) -> String {
        format!(
            "{}/deployments/{}/embeddings?api-version={}",
            self.base_url, deployment, self.api_version
        )
    }

    /// Convert OpenAI format to Azure AI format
    fn convert_request(&self, request: ChatCompletionRequest) -> AzureAiRequest {
        AzureAiRequest {
            messages: request
                .messages
                .into_iter()
                .map(|msg| match msg {
                    llmg_core::types::Message::User { content, .. } => AzureAiMessage {
                        role: "user".to_string(),
                        content,
                    },
                    llmg_core::types::Message::Assistant { content, .. } => AzureAiMessage {
                        role: "assistant".to_string(),
                        content: content.unwrap_or_default(),
                    },
                    llmg_core::types::Message::System { content, .. } => AzureAiMessage {
                        role: "system".to_string(),
                        content,
                    },
                    _ => AzureAiMessage {
                        role: "user".to_string(),
                        content: String::new(),
                    },
                })
                .collect(),
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: request.stream,
        }
    }

    /// Convert Azure AI response to OpenAI format
    fn convert_response(&self, response: AzureAiResponse) -> ChatCompletionResponse {
        let usage = response.usage.map(|u| llmg_core::types::Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });

        ChatCompletionResponse {
            id: response.id,
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
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
            usage,
        }
    }

    async fn make_request(
        &self,
        request: ChatCompletionRequest,
        deployment: &str,
    ) -> Result<ChatCompletionResponse, LlmError> {
        let azure_req = self.convert_request(request.clone());
        let url = self.build_url(deployment);

        let mut req = self
            .http_client
            .post(&url)
            .json(&azure_req)
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

        let azure_resp: AzureAiResponse = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        Ok(self.convert_response(azure_resp))
    }
}

#[async_trait::async_trait]
impl Provider for AzureAiClient {
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        let deployment = request
            .model
            .split('/')
            .next_back()
            .ok_or_else(|| {
                LlmError::InvalidRequest(
                    "Invalid model format: expected provider/model".to_string(),
                )
            })?
            .to_string();
        self.make_request(request, &deployment).await
    }

    async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        let deployment = request
            .model
            .split('/')
            .next_back()
            .unwrap_or(&request.model)
            .to_string();
        let url = self.build_embeddings_url(&deployment);

        let body = serde_json::json!({
            "input": request.input,
            "model": deployment,
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

        let resp: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        let data = resp["data"]
            .as_array()
            .ok_or_else(|| LlmError::ProviderError("Missing data in response".to_string()))?;

        let embeddings: Vec<Embedding> = data
            .iter()
            .enumerate()
            .map(|(i, item)| Embedding {
                index: i as u32,
                object: "embedding".to_string(),
                embedding: item["embedding"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_f64().map(|f| f as f32))
                            .collect()
                    })
                    .unwrap_or_default(),
            })
            .collect();

        let prompt_tokens = resp["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32;
        let total_tokens = resp["usage"]["total_tokens"].as_u64().unwrap_or(0) as u32;

        Ok(EmbeddingResponse {
            id: format!("azure-emb-{}", uuid::Uuid::new_v4()),
            object: "list".to_string(),
            data: embeddings,
            model: deployment,
            usage: Usage {
                prompt_tokens,
                completion_tokens: 0,
                total_tokens,
            },
        })
    }
    fn provider_name(&self) -> &'static str {
        "azure_ai"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_azure_ai_client_creation() {
        let client = AzureAiClient::new(
            "test-key",
            "https://api.azure.microsoft.com",
            Some("test-project".to_string()),
        );
        assert_eq!(client.provider_name(), "azure_ai");
    }

    #[test]
    fn test_url_building() {
        let client = AzureAiClient::new("test-key", "https://api.azure.microsoft.com", None);
        let url = client.build_url("my-deployment");
        assert!(url.contains("api.azure.microsoft.com"));
        assert!(url.contains("my-deployment"));
    }
}

