use llmg_core::{
    provider::{ApiKeyCredentials, Credentials, LlmError, Provider},
    types::{ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse},
};

/// Azure OpenAI API client
#[derive(Debug)]
pub struct AzureOpenAiClient {
    http_client: reqwest::Client,
    endpoint: String,
    deployment: String,
    credentials: Box<dyn Credentials>,
    api_version: String,
}

impl AzureOpenAiClient {
    /// Create a new Azure OpenAI client from environment
    ///
    /// Expects:
    /// - AZURE_OPENAI_API_KEY: The API key
    /// - AZURE_OPENAI_ENDPOINT: The endpoint URL (e.g., https://my-resource.openai.azure.com)
    /// - AZURE_OPENAI_DEPLOYMENT: The deployment name
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("AZURE_OPENAI_API_KEY").map_err(|_| LlmError::AuthError)?;
        let endpoint = std::env::var("AZURE_OPENAI_ENDPOINT")
            .map_err(|_| LlmError::InvalidRequest("AZURE_OPENAI_ENDPOINT not set".to_string()))?;
        let deployment = std::env::var("AZURE_OPENAI_DEPLOYMENT")
            .map_err(|_| LlmError::InvalidRequest("AZURE_OPENAI_DEPLOYMENT not set".to_string()))?;

        Ok(Self::new(api_key, endpoint, deployment))
    }

    /// Create a new Azure OpenAI client with explicit configuration
    pub fn new(
        api_key: impl Into<String>,
        endpoint: impl Into<String>,
        deployment: impl Into<String>,
    ) -> Self {
        let api_key = api_key.into();
        let endpoint = endpoint.into();
        let deployment = deployment.into();

        Self {
            http_client: reqwest::Client::new(),
            endpoint,
            deployment,
            credentials: Box::new(ApiKeyCredentials::with_header(api_key, "api-key")),
            api_version: "2024-02-01".to_string(),
        }
    }

    /// Set a custom API version
    pub fn with_api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = version.into();
        self
    }

    /// Build the chat completions URL
    fn build_url(&self) -> String {
        format!(
            "{}/openai/deployments/{}/chat/completions?api-version={}",
            self.endpoint, self.deployment, self.api_version
        )
    }

    /// Map model name to deployment
    ///
    /// In Azure, the deployment name is separate from the model name
    pub fn map_model_to_deployment(&self, _model: &str) -> String {
        // For Azure, we typically use the deployment name directly
        // but this method allows for mapping if needed
        self.deployment.clone()
    }

    async fn make_request(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        let url = self.build_url();

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

        let mut chat_response: ChatCompletionResponse = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        // Override the model name with the requested one
        chat_response.model = request.model;

        Ok(chat_response)
    }
}

#[async_trait::async_trait]
impl Provider for AzureOpenAiClient {
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        self.make_request(request).await
    }

    async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        let url = format!(
            "{}/openai/deployments/{}/embeddings?api-version={}",
            self.endpoint, self.deployment, self.api_version
        );

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
        "azure"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_azure_client_creation() {
        let client = AzureOpenAiClient::new(
            "test-key",
            "https://my-resource.openai.azure.com",
            "my-deployment",
        );

        assert_eq!(client.provider_name(), "azure");
    }

    #[test]
    fn test_url_building() {
        let client = AzureOpenAiClient::new(
            "test-key",
            "https://my-resource.openai.azure.com",
            "my-deployment",
        );

        let url = client.build_url();
        assert!(url.contains("openai.azure.com"));
        assert!(url.contains("my-deployment"));
        assert!(url.contains("api-version=2024-02-01"));
    }

    #[test]
    fn test_custom_api_version() {
        let client = AzureOpenAiClient::new(
            "test-key",
            "https://my-resource.openai.azure.com",
            "my-deployment",
        )
        .with_api_version("2023-05-15");

        let url = client.build_url();
        assert!(url.contains("api-version=2023-05-15"));
    }
}

