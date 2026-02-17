use llmg_core::{
    provider::{Credentials, LlmError, Provider},
    types::{
        ChatCompletionRequest, ChatCompletionResponse, Embedding, EmbeddingRequest,
        EmbeddingResponse, Usage,
    },
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// AWS SageMaker API client
#[derive(Debug)]
pub struct AwsSagemakerClient {
    http_client: reqwest::Client,
    endpoint: String,
    credentials: Box<dyn Credentials>,
    region: String,
    model_id: Option<String>,
    session_token: Option<String>,
}

/// AWS SigV4 signature context
#[derive(Debug)]
struct SigV4Context {
    access_key: String,
    secret_key: String,
    region: String,
    service: String,
}

/// SageMaker embedding request (HuggingFace-style)
#[derive(Debug, Serialize)]
struct SageMakerEmbeddingRequest {
    inputs: String,
}

/// SageMaker request format
#[derive(Debug, Serialize)]
struct SageMakerRequest {
    inputs: String,
    parameters: Option<SageMakerParameters>,
}

/// SageMaker inference parameters
#[derive(Debug, Serialize)]
struct SageMakerParameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_new_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
}

/// SageMaker response format
#[derive(Debug, Deserialize)]
struct SageMakerResponse {
    #[serde(default)]
    outputs: Vec<SageMakerOutput>,
}

#[derive(Debug, Deserialize)]
struct SageMakerOutput {
    #[serde(rename = "generated_text")]
    generated_text: String,
}

impl AwsSagemakerClient {
    /// Create a new AWS SageMaker client from environment
    pub fn from_env() -> Result<Self, LlmError> {
        let access_key = std::env::var("AWS_ACCESS_KEY_ID").map_err(|_| LlmError::AuthError)?;
        let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY").map_err(|_| LlmError::AuthError)?;
        let endpoint = std::env::var("AWS_SAGEMAKER_ENDPOINT")
            .unwrap_or_else(|_| "https://api.sagemaker.aws.amazon.com".to_string());
        let region = std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string());
        let model_id = std::env::var("AWS_SAGEMAKER_MODEL_ID").ok();
        let session_token = std::env::var("AWS_SESSION_TOKEN").ok();

        Ok(Self::new(
            access_key,
            secret_key,
            endpoint,
            region.as_str(),
            model_id,
            session_token,
        ))
    }

    /// Create a new AWS SageMaker client with explicit configuration
    pub fn new(
        access_key: impl Into<String>,
        secret_key: impl Into<String>,
        endpoint: impl Into<String>,
        region: &str,
        model_id: Option<String>,
        session_token: Option<String>,
    ) -> Self {
        let sigv4_context = SigV4Context {
            access_key: access_key.into(),
            secret_key: secret_key.into(),
            region: region.to_string(),
            service: "sagemaker".to_string(),
        };

        Self {
            http_client: reqwest::Client::new(),
            endpoint: endpoint.into(),
            credentials: Box::new(SigV4Credentials::new(sigv4_context)),
            region: region.to_string(),
            model_id,
            session_token,
        }
    }

    /// Set a custom model ID
    pub fn with_model_id(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = Some(model_id.into());
        self
    }

    /// Set a custom region
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = region.into();
        self
    }

    /// Build inference URL
    fn build_url(&self, model: &str) -> String {
        format!("{}/endpoints/{}/invocations", self.endpoint, model)
    }

    /// Convert OpenAI format to SageMaker format
    fn convert_request(&self, request: &ChatCompletionRequest) -> SageMakerRequest {
        let content = request
            .messages
            .iter()
            .map(|msg| match msg {
                llmg_core::types::Message::User { content, .. } => format!("User: {}", content),
                llmg_core::types::Message::Assistant { content, .. } => {
                    format!("Assistant: {}", content.as_deref().unwrap_or(""))
                }
                llmg_core::types::Message::System { content, .. } => {
                    format!("System: {}", content)
                }
                _ => String::new(),
            })
            .collect::<Vec<_>>()
            .join("\n");

        SageMakerRequest {
            inputs: content,
            parameters: Some(SageMakerParameters {
                max_new_tokens: request.max_tokens,
                temperature: request.temperature,
                top_p: request.top_p,
            }),
        }
    }

    /// Convert SageMaker response to OpenAI format
    fn convert_response(
        &self,
        response: SageMakerResponse,
        model: String,
    ) -> ChatCompletionResponse {
        let generated_text = response
            .outputs
            .first()
            .map(|o| o.generated_text.clone())
            .unwrap_or_default();

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
            usage: None,
        }
    }

    async fn make_request(
        &self,
        request: &ChatCompletionRequest,
        model: &str,
    ) -> Result<ChatCompletionResponse, LlmError> {
        let sm_req = self.convert_request(request);
        let url = self.build_url(model);

        let mut req = self
            .http_client
            .post(&url)
            .json(&sm_req)
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

        let sm_resp: SageMakerResponse = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        Ok(self.convert_response(sm_resp, model.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aws_sagemaker_client_creation() {
        let client = AwsSagemakerClient::new(
            "test-key",
            "test-secret",
            "https://api.sagemaker.aws.amazon.com",
            "us-east-1",
            None,
            None,
        );
        assert_eq!(client.provider_name(), "aws_sagemaker");
    }

    #[test]
    fn test_url_building() {
        let client = AwsSagemakerClient::new(
            "test-key",
            "test-secret",
            "https://api.sagemaker.aws.amazon.com",
            "us-east-1",
            None,
            None,
        );
        let url = client.build_url("test-endpoint");
        assert!(url.contains("api.sagemaker.aws.amazon.com"));
        assert!(url.contains("test-endpoint"));
        assert!(url.contains("invocations"));
    }
}

/// AWS SigV4 credentials implementation
#[derive(Debug)]
struct SigV4Credentials {
    context: SigV4Context,
}

impl SigV4Credentials {
    fn new(context: SigV4Context) -> Self {
        Self { context }
    }

    fn get_headers(
        &self,
        _request: &mut reqwest::Request,
    ) -> Result<HashMap<String, String>, LlmError> {
        // Placeholder for SigV4 signature generation
        // In production, this would:
        // 1. Extract request method, URI, headers, and body
        // 2. Create canonical request
        // 3. Create string to sign
        // 4. Calculate signature using HMAC-SHA256
        // 5. Add Authorization header with signature

        let mut headers = HashMap::new();
        headers.insert(
            "x-amz-date".to_string(),
            chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string(),
        );

        let token = &self.context.access_key;
        headers.insert(
            "Authorization".to_string(),
            format!(
                "AWS4-HMAC-SHA256 Credential={}/{}/{}/{}/aws4_request",
                token,
                self.context.region,
                "sagemaker",
                chrono::Utc::now().format("%Y%m%d")
            ),
        );

        Ok(headers)
    }
}

impl Credentials for SigV4Credentials {
    fn apply(&self, request: &mut reqwest::Request) -> Result<(), LlmError> {
        let headers = self.get_headers(request)?;
        for (key, value) in headers {
            if let Ok(header_name) = reqwest::header::HeaderName::from_bytes(key.as_bytes()) {
                request
                    .headers_mut()
                    .insert(header_name, value.parse().unwrap());
            }
        }
        Ok(())
    }
}

#[async_trait::async_trait]
impl Provider for AwsSagemakerClient {
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        let model = if let Some(mid) = &self.model_id {
            mid.as_str()
        } else {
            request
                .model
                .split('/')
                .next_back()
                .unwrap_or(&request.model)
        };
        self.make_request(&request, model).await
    }

    async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        let model = if let Some(mid) = &self.model_id {
            mid.clone()
        } else {
            request
                .model
                .split('/')
                .next_back()
                .unwrap_or(&request.model)
                .to_string()
        };

        let url = self.build_url(&model);

        let embed_req = SageMakerEmbeddingRequest {
            inputs: request.input,
        };

        let mut req = self
            .http_client
            .post(&url)
            .json(&embed_req)
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

        // HuggingFace-style response: [[0.1, 0.2, ...]]
        let raw: Vec<Vec<f32>> = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        let embeddings: Vec<Embedding> = raw
            .into_iter()
            .enumerate()
            .map(|(i, values)| Embedding {
                index: i as u32,
                object: "embedding".to_string(),
                embedding: values,
            })
            .collect();

        Ok(EmbeddingResponse {
            id: format!("sagemaker-emb-{}", uuid::Uuid::new_v4()),
            object: "list".to_string(),
            data: embeddings,
            model,
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
        })
    }
    fn provider_name(&self) -> &'static str {
        "aws_sagemaker"
    }
}

/*#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aws_sagemaker_client_creation() {
        let client = AwsSagemakerClient::new(
            "test-key",
            "test-secret",
            "https://api.sagemaker.aws.amazon.com",
            "us-east-1",
            None,
            None,
        );
        assert_eq!(client.provider_name(), "aws_sagemaker");
    }

    #[test]
    fn test_url_building() {
        let client = AwsSagemakerClient::new(
            "test-key",
            "test-secret",
            "https://api.sagemaker.aws.amazon.com",
            "us-east-1",
            None,
            None,
        );
        let url = client.build_url("test-endpoint");
        assert!(url.contains("api.sagemaker.aws.amazon.com"));
        assert!(url.contains("test-endpoint"));
        assert!(url.contains("invocations"));
    }
*/
