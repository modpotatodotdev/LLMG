use llmg_core::provider::{Credentials, LlmError, Provider};
use llmg_core::types::{
    ChatCompletionRequest, ChatCompletionResponse, Embedding, EmbeddingRequest, EmbeddingResponse,
    Usage,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// AWS Bedrock API client
#[derive(Debug)]
pub struct BedrockClient {
    http_client: reqwest::Client,
    base_url: String,
    credentials: Box<dyn Credentials>,
    region: String,
    model_providers: HashMap<String, BedrockModelProvider>,
    session_token: Option<String>,
}

/// Bedrock model provider information
#[derive(Debug, Clone)]
struct BedrockModelProvider {
    provider: String,
    model_id: String,
}

/// Bedrock request format
#[derive(Debug, Serialize)]
struct BedrockRequest {
    anthropic_version: Option<String>,
    claude: Option<AnthropicPayload>,
    amazon: Option<AmazonPayload>,
    ai21: Option<Ai21Payload>,
    cohere: Option<CoherePayload>,
    meta: Option<MetaPayload>,
}

#[derive(Debug, Serialize)]
struct AnthropicPayload {
    messages: Vec<BedrockMessage>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
#[allow(non_snake_case)]
struct AmazonPayload {
    inputText: String,
    textGenerationConfig: Option<AmazonGenerationConfig>,
}

#[derive(Debug, Serialize)]
#[allow(non_snake_case)]
struct AmazonGenerationConfig {
    maxTokenCount: Option<u32>,
    temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
#[allow(non_snake_case)]
struct Ai21Payload {
    prompt: String,
    maxTokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
struct CoherePayload {
    prompt: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
struct MetaPayload {
    prompt: String,
    max_gen_len: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

/// Bedrock Titan embedding request
#[derive(Debug, Serialize)]
struct BedrockEmbeddingRequest {
    #[serde(rename = "inputText")]
    input_text: String,
}

/// Bedrock Titan embedding response
#[derive(Debug, Deserialize)]
struct BedrockEmbeddingResponse {
    embedding: Vec<f32>,
    #[serde(rename = "inputTextTokenCount")]
    input_text_token_count: Option<u32>,
}

/// Bedrock message format
#[derive(Debug, Serialize)]
struct BedrockMessage {
    role: String,
    content: String,
}

/// Bedrock response format
#[derive(Debug, Deserialize)]
struct BedrockResponse {
    #[serde(default)]
    completion: String,
    #[serde(default)]
    message: BedrockResponseMessage,
}

#[derive(Debug, Default, Deserialize)]
struct BedrockResponseMessage {
    #[serde(default)]
    content: Vec<BedrockContent>,
}

#[derive(Debug, Deserialize)]
struct BedrockContent {
    #[serde(default)]
    text: String,
}

/// Bedrock response wrapper
#[derive(Debug, Deserialize)]
struct BedrockResponseWrapper {
    #[serde(default)]
    output: Option<BedrockResponseOutput>,
}

#[derive(Debug, Deserialize)]
struct BedrockResponseOutput {
    #[serde(rename = "completion", default)]
    completion: String,
    #[serde(rename = "message", default)]
    message: BedrockResponseMessage,
}

impl BedrockClient {
    /// Create a new AWS Bedrock client from environment
    pub fn from_env() -> Result<Self, LlmError> {
        let access_key = std::env::var("AWS_ACCESS_KEY_ID").map_err(|_| LlmError::AuthError)?;
        let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY").map_err(|_| LlmError::AuthError)?;
        let region = std::env::var("AWS_REGION").unwrap_or_else(|_| "us-west-2".to_string());
        let session_token = std::env::var("AWS_SESSION_TOKEN").ok();

        Ok(Self::new(access_key, secret_key, region, session_token))
    }

    /// Create a new AWS Bedrock client with explicit configuration
    pub fn new(
        access_key: impl Into<String>,
        secret_key: impl Into<String>,
        region: impl Into<String>,
        session_token: Option<String>,
    ) -> Self {
        let region = region.into();

        let sigv4_context = SigV4Context {
            access_key: access_key.into(),
            secret_key: secret_key.into(),
            region: region.clone(),
            service: "bedrock".to_string(),
        };

        let mut model_providers = HashMap::new();

        // Anthropic Claude models
        model_providers.insert(
            "anthropic.claude-3-opus-20240229-v1:0".to_string(),
            BedrockModelProvider {
                provider: "anthropic".to_string(),
                model_id: "anthropic.claude-3-opus-20240229-v1:0".to_string(),
            },
        );
        model_providers.insert(
            "anthropic.claude-3-sonnet-20240229-v1:0".to_string(),
            BedrockModelProvider {
                provider: "anthropic".to_string(),
                model_id: "anthropic.claude-3-sonnet-20240229-v1:0".to_string(),
            },
        );
        model_providers.insert(
            "anthropic.claude-3-haiku-20240307-v1:0".to_string(),
            BedrockModelProvider {
                provider: "anthropic".to_string(),
                model_id: "anthropic.claude-3-haiku-20240307-v1:0".to_string(),
            },
        );

        // Amazon Titan models
        model_providers.insert(
            "amazon.titan-text-express-v1".to_string(),
            BedrockModelProvider {
                provider: "amazon".to_string(),
                model_id: "amazon.titan-text-express-v1".to_string(),
            },
        );
        model_providers.insert(
            "amazon.titan-text-premier-v1:0".to_string(),
            BedrockModelProvider {
                provider: "amazon".to_string(),
                model_id: "amazon.titan-text-premier-v1:0".to_string(),
            },
        );

        // AI21 Jurassic models
        model_providers.insert(
            "ai21.j2-ultra-v1".to_string(),
            BedrockModelProvider {
                provider: "ai21".to_string(),
                model_id: "ai21.j2-ultra-v1".to_string(),
            },
        );
        model_providers.insert(
            "ai21.jamba-1-5-large-v1:0".to_string(),
            BedrockModelProvider {
                provider: "ai21".to_string(),
                model_id: "ai21.jamba-1-5-large-v1:0".to_string(),
            },
        );

        // Cohere Command models
        model_providers.insert(
            "cohere.command-text-v14".to_string(),
            BedrockModelProvider {
                provider: "cohere".to_string(),
                model_id: "cohere.command-text-v14".to_string(),
            },
        );
        model_providers.insert(
            "cohere.command-r-plus-v1:0".to_string(),
            BedrockModelProvider {
                provider: "cohere".to_string(),
                model_id: "cohere.command-r-plus-v1:0".to_string(),
            },
        );

        // Meta Llama models
        model_providers.insert(
            "meta.llama2-13b-chat-v1".to_string(),
            BedrockModelProvider {
                provider: "meta".to_string(),
                model_id: "meta.llama2-13b-chat-v1".to_string(),
            },
        );
        model_providers.insert(
            "meta.llama3-8b-instruct-v1:0".to_string(),
            BedrockModelProvider {
                provider: "meta".to_string(),
                model_id: "meta.llama3-8b-instruct-v1:0".to_string(),
            },
        );
        model_providers.insert(
            "meta.llama3-70b-instruct-v1:0".to_string(),
            BedrockModelProvider {
                provider: "meta".to_string(),
                model_id: "meta.llama3-70b-instruct-v1:0".to_string(),
            },
        );

        Self {
            http_client: reqwest::Client::new(),
            base_url: format!("https://bedrock-runtime.{}.amazonaws.com", region),
            credentials: Box::new(BedrockSigV4Credentials::new(sigv4_context)),
            region,
            model_providers,
            session_token,
        }
    }

    /// Set a custom region
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = region.into();
        self.base_url = format!("https://bedrock-runtime.{}.amazonaws.com", self.region);
        self
    }

    /// Get model provider information
    fn get_model_provider(&self, model: &str) -> Option<&BedrockModelProvider> {
        // Try direct match first
        if let Some(provider) = self.model_providers.get(model) {
            return Some(provider);
        }

        // Try removing provider prefix
        let parts: Vec<&str> = model.split('/').collect();
        if parts.len() > 1 {
            return self.model_providers.get(parts[1]);
        }

        // Try with bedrock prefix
        let bedrock_key = format!("bedrock/{}", model);
        self.model_providers.get(&bedrock_key)
    }

    /// Build inference URL
    fn build_url(&self, model_id: &str) -> String {
        format!(
            "{}/model/{}:invoke-with-response-stream",
            self.base_url, model_id
        )
    }

    /// Convert OpenAI format to Bedrock format based on provider
    fn convert_request(&self, request: ChatCompletionRequest, provider: &str) -> BedrockRequest {
        let content = request
            .messages
            .iter()
            .filter_map(|msg| match msg {
                llmg_core::types::Message::User { content, .. } => Some(BedrockMessage {
                    role: "user".to_string(),
                    content: content.clone(),
                }),
                llmg_core::types::Message::Assistant { content, .. } => {
                    content.as_ref().map(|c| BedrockMessage {
                        role: "assistant".to_string(),
                        content: c.clone(),
                    })
                }
                llmg_core::types::Message::System { content, .. } => Some(BedrockMessage {
                    role: "user".to_string(),
                    content: format!("System: {}", content),
                }),
                _ => None,
            })
            .collect::<Vec<_>>();

        let max_tokens = request.max_tokens.unwrap_or(1000);
        let temperature = request.temperature;

        match provider {
            "anthropic" => BedrockRequest {
                anthropic_version: Some("bedrock-2023-05-31".to_string()),
                claude: Some(AnthropicPayload {
                    messages: content,
                    max_tokens,
                    temperature,
                }),
                amazon: None,
                ai21: None,
                cohere: None,
                meta: None,
            },
            "amazon" => {
                let prompt = content
                    .iter()
                    .map(|m| format!("{}: {}", m.role, m.content))
                    .collect::<Vec<_>>()
                    .join("\n");

                BedrockRequest {
                    anthropic_version: None,
                    claude: None,
                    amazon: Some(AmazonPayload {
                        inputText: prompt,
                        textGenerationConfig: Some(AmazonGenerationConfig {
                            maxTokenCount: Some(max_tokens),
                            temperature,
                        }),
                    }),
                    ai21: None,
                    cohere: None,
                    meta: None,
                }
            }
            "ai21" => {
                let prompt = content
                    .iter()
                    .map(|m| m.content.clone())
                    .collect::<Vec<_>>()
                    .join("\n");

                BedrockRequest {
                    anthropic_version: None,
                    claude: None,
                    amazon: None,
                    ai21: Some(Ai21Payload {
                        prompt,
                        maxTokens: max_tokens,
                        temperature,
                    }),
                    cohere: None,
                    meta: None,
                }
            }
            "cohere" => {
                let prompt = content
                    .iter()
                    .map(|m| m.content.clone())
                    .collect::<Vec<_>>()
                    .join("\n");

                BedrockRequest {
                    anthropic_version: None,
                    claude: None,
                    amazon: None,
                    ai21: None,
                    cohere: Some(CoherePayload {
                        prompt,
                        max_tokens,
                        temperature,
                    }),
                    meta: None,
                }
            }
            "meta" => {
                let prompt = content
                    .iter()
                    .map(|m| format!("{}: {}", m.role, m.content))
                    .collect::<Vec<_>>()
                    .join("\n");

                BedrockRequest {
                    anthropic_version: None,
                    claude: None,
                    amazon: None,
                    ai21: None,
                    cohere: None,
                    meta: Some(MetaPayload {
                        prompt,
                        max_gen_len: max_tokens,
                        temperature,
                    }),
                }
            }
            _ => BedrockRequest {
                anthropic_version: None,
                claude: None,
                amazon: None,
                ai21: None,
                cohere: None,
                meta: None,
            },
        }
    }

    /// Convert Bedrock response to OpenAI format
    fn convert_response(
        &self,
        response: BedrockResponseWrapper,
        model: String,
    ) -> ChatCompletionResponse {
        let text = response
            .output
            .and_then(|o| {
                if !o.completion.is_empty() {
                    Some(o.completion)
                } else {
                    o.message.content.first().map(|c| c.text.clone())
                }
            })
            .unwrap_or_default();

        ChatCompletionResponse {
            id: uuid::Uuid::new_v4().to_string(),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            choices: vec![llmg_core::types::Choice {
                index: 0,
                message: llmg_core::types::Message::Assistant {
                    content: Some(text),
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
        let model_provider = self
            .get_model_provider(model)
            .ok_or_else(|| LlmError::InvalidRequest(format!("Unknown model: {}", model)))?;

        let bedrock_req = self.convert_request(request.clone(), &model_provider.provider);
        let url = self.build_url(&model_provider.model_id);

        let mut req = self
            .http_client
            .post(&url)
            .json(&bedrock_req)
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

        let bedrock_resp: BedrockResponseWrapper = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        Ok(self.convert_response(bedrock_resp, model.to_string()))
    }
}

/// AWS SigV4 signature context
#[derive(Debug)]
struct SigV4Context {
    access_key: String,
    secret_key: String,
    region: String,
    service: String,
}

/// Bedrock SigV4 credentials implementation
#[derive(Debug)]
struct BedrockSigV4Credentials {
    context: SigV4Context,
}

impl BedrockSigV4Credentials {
    fn new(context: SigV4Context) -> Self {
        Self { context }
    }

    fn get_headers(
        &self,
        _request: &mut reqwest::Request,
    ) -> Result<HashMap<String, String>, LlmError> {
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
                chrono::Utc::now().format("%Y%m%d"),
                self.context.region,
                self.context.service
            ),
        );

        Ok(headers)
    }
}

impl Credentials for BedrockSigV4Credentials {
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
impl Provider for BedrockClient {
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        self.make_request(&request, &request.model).await
    }

    async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        let model_id = request
            .model
            .split('/')
            .next_back()
            .unwrap_or(&request.model)
            .to_string();

        let embed_req = BedrockEmbeddingRequest {
            input_text: request.input,
        };

        let url = format!("{}/model/{}/invoke", self.base_url, model_id);

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

        let embed_resp: BedrockEmbeddingResponse = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        let token_count = embed_resp.input_text_token_count.unwrap_or(0);

        Ok(EmbeddingResponse {
            id: format!("bedrock-emb-{}", uuid::Uuid::new_v4()),
            object: "list".to_string(),
            data: vec![Embedding {
                index: 0,
                object: "embedding".to_string(),
                embedding: embed_resp.embedding,
            }],
            model: model_id,
            usage: Usage {
                prompt_tokens: token_count,
                completion_tokens: 0,
                total_tokens: token_count,
            },
        })
    }
    fn provider_name(&self) -> &'static str {
        "bedrock"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bedrock_client_creation() {
        let client = BedrockClient::new("test-key", "test-secret", "us-west-2", None);
        assert_eq!(client.provider_name(), "bedrock");
    }

    #[test]
    fn test_model_provider_lookup() {
        let client = BedrockClient::new("test-key", "test-secret", "us-west-2", None);
        let provider = client.get_model_provider("anthropic.claude-3-opus-20240229-v1:0");
        assert!(provider.is_some());
        assert_eq!(provider.unwrap().provider, "anthropic");
    }

    #[test]
    fn test_url_building() {
        let client = BedrockClient::new("test-key", "test-secret", "us-west-2", None);
        let url = client.build_url("anthropic.claude-3-opus-20240229-v1:0");
        assert!(url.contains("bedrock-runtime.us-west-2.amazonaws.com"));
        assert!(url.contains("anthropic.claude-3-opus-20240229-v1:0"));
    }
}

