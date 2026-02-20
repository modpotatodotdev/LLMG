//! GitHub Copilot API client for LLMG
//!
//! Implements the Provider trait for GitHub Copilot Chat API using OAuth device flow.
//!
//! Authentication Flow:
//! 1. Get device code from GitHub
//! 2. User visits verification URL and enters code
//! 3. Poll for access token
//! 4. Exchange access token for Copilot API key
//!
//! Environment variables:
//! - GITHUB_COPILOT_ACCESS_TOKEN: Cached OAuth access token
//! - GITHUB_COPILOT_API_KEY: Cached Copilot API key
//! - GITHUB_COPILOT_TOKEN_DIR: Directory to store tokens (default: ~/.config/llmg/github_copilot)

use llmg_core::{
    provider::{LlmError, Provider},
    types::{
        ChatCompletionRequest, ChatCompletionResponse, Choice, EmbeddingRequest, EmbeddingResponse,
        Message, Usage,
    },
};
// use serde::Deserialize; // removed unused import
use std::collections::HashMap;
use std::path::PathBuf;

/// GitHub Copilot OAuth constants
const GITHUB_CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";
const GITHUB_DEVICE_CODE_URL: &str = "https://github.com/login/device/code";
const GITHUB_ACCESS_TOKEN_URL: &str = "https://github.com/login/oauth/access_token";
const GITHUB_COPILOT_TOKEN_URL: &str = "https://api.github.com/copilot_internal/v2/token";
const GITHUB_COPILOT_API_BASE: &str = "https://api.githubcopilot.com";

/// GitHub Copilot API client
#[derive(Debug, Clone)]
pub struct GitHubCopilotClient {
    http_client: reqwest::Client,
    api_key: String,
    access_token: String,
    editor_version: String,
    integration_id: String,
}

#[derive(Debug, serde::Deserialize)]
struct DeviceCodeResponse {
    device_code: String,
    user_code: String,
    verification_uri: String,
    expires_in: i32,
    interval: i32,
}

#[derive(Debug, serde::Deserialize)]
struct AccessTokenResponse {
    access_token: Option<String>,
    token_type: Option<String>,
    error: Option<String>,
    error_description: Option<String>,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct CopilotApiKeyResponse {
    token: String,
    expires_at: i64,
    endpoints: Option<HashMap<String, String>>,
}

#[derive(Debug, serde::Serialize)]
struct CopilotChatRequest {
    messages: Vec<CopilotMessage>,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Debug, serde::Serialize)]
struct CopilotMessage {
    role: String,
    content: String,
}

#[derive(Debug, serde::Deserialize)]
struct CopilotChatResponse {
    id: String,
    #[serde(default)]
    object: String,
    #[serde(default)]
    created: i64,
    model: String,
    choices: Vec<CopilotChoice>,
    usage: Option<CopilotUsage>,
}

#[derive(Debug, serde::Deserialize)]
struct CopilotChoice {
    index: i32,
    message: CopilotMessageResponse,
    #[serde(rename = "finish_reason")]
    finish_reason: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct CopilotMessageResponse {
    role: String,
    content: String,
}

#[derive(Debug, serde::Deserialize)]
struct CopilotUsage {
    #[serde(rename = "prompt_tokens")]
    prompt_tokens: u32,
    #[serde(default, rename = "completion_tokens")]
    completion_tokens: u32,
    #[serde(rename = "total_tokens")]
    total_tokens: u32,
}

#[derive(Debug, serde::Serialize)]
struct CopilotEmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Debug, serde::Deserialize)]
struct CopilotEmbeddingResponse {
    #[serde(default)]
    object: String,
    data: Vec<CopilotEmbeddingData>,
    #[serde(default)]
    model: String,
    usage: CopilotUsage,
}

#[derive(Debug, serde::Deserialize)]
struct CopilotEmbeddingData {
    object: String,
    index: u32,
    embedding: Vec<f32>,
}

impl GitHubCopilotClient {
    /// Create a new GitHub Copilot client with OAuth flow
    pub async fn new() -> Result<Self, LlmError> {
        let token_dir = Self::get_token_dir();
        std::fs::create_dir_all(&token_dir).map_err(|e| {
            LlmError::ProviderError(format!("Failed to create token directory: {}", e))
        })?;

        println!("Loading access token...");
        let access_token = match Self::load_cached_access_token(&token_dir).await {
            Ok(token) => token,
            Err(e) => {
                println!("load_cached_access_token failed with {:?}", e);
                return Err(e);
            }
        };
        println!("Access token loaded.");

        let mut client = Self {
            http_client: reqwest::Client::new(),
            api_key: String::new(),
            access_token,
            editor_version: "vscode/1.85.1".to_string(),
            integration_id: "vscode-chat".to_string(),
        };

        if let Ok(key) = Self::load_cached_api_key(&token_dir).await {
            client.api_key = key;
            println!("API key loaded from cache.");
        } else {
            println!("Refreshing API key...");
            match client.refresh_api_key().await {
                Ok(_) => println!("API key refreshed."),
                Err(e) => {
                    println!("refresh_api_key failed with {:?}", e);
                    return Err(e);
                }
            }
        }

        Ok(client)
    }

    /// Create with explicit API key (for testing or pre-authenticated scenarios)
    pub fn with_api_key(api_key: impl Into<String>, access_token: impl Into<String>) -> Self {
        Self {
            http_client: reqwest::Client::new(),
            api_key: api_key.into(),
            access_token: access_token.into(),
            editor_version: "vscode/1.85.1".to_string(),
            integration_id: "vscode-chat".to_string(),
        }
    }

    fn get_token_dir() -> PathBuf {
        std::env::var("GITHUB_COPILOT_TOKEN_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                dirs::config_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join("llmg/github_copilot")
            })
    }

    async fn load_cached_access_token(token_dir: &std::path::Path) -> Result<String, LlmError> {
        let access_token_path = token_dir.join("access-token");

        if let Ok(token) = std::fs::read_to_string(&access_token_path) {
            let token = token.trim();
            if !token.is_empty() {
                return Ok(token.to_string());
            }
        }

        Self::perform_oauth_flow(token_dir).await
    }

    async fn load_cached_api_key(token_dir: &std::path::Path) -> Result<String, LlmError> {
        let api_key_path = token_dir.join("api-key.json");

        if let Ok(content) = std::fs::read_to_string(&api_key_path) {
            if let Ok(api_key_info) = serde_json::from_str::<CopilotApiKeyResponse>(&content) {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as i64;

                if api_key_info.expires_at > now {
                    return Ok(api_key_info.token);
                }
            }
        }

        Err(LlmError::AuthError)
    }

    async fn perform_oauth_flow(token_dir: &std::path::Path) -> Result<String, LlmError> {
        let device_code_resp = Self::get_device_code().await?;

        eprintln!("\n🔐 GitHub Copilot Authentication Required");
        eprintln!("Please visit: {}", device_code_resp.verification_uri);
        eprintln!("And enter code: {}\n", device_code_resp.user_code);

        let access_token = Self::poll_for_access_token(
            &device_code_resp.device_code,
            device_code_resp.interval as u64,
        )
        .await?;

        let access_token_path = token_dir.join("access-token");
        std::fs::write(&access_token_path, &access_token)
            .map_err(|e| LlmError::ProviderError(format!("Failed to cache access token: {}", e)))?;

        Ok(access_token)
    }

    async fn get_device_code() -> Result<DeviceCodeResponse, LlmError> {
        let client = reqwest::Client::new();

        let resp = client
            .post(GITHUB_DEVICE_CODE_URL)
            .header("Accept", "application/json")
            .header("User-Agent", "GithubCopilot/1.155.0")
            .json(&serde_json::json!({
                "client_id": GITHUB_CLIENT_ID,
                "scope": "read:user"
            }))
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("Failed to get device code: {}", e)))?;

        if !resp.status().is_success() {
            return Err(LlmError::ApiError {
                status: resp.status().as_u16(),
                message: resp.text().await.unwrap_or_default(),
            });
        }

        resp.json::<DeviceCodeResponse>()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))
    }

    async fn poll_for_access_token(device_code: &str, interval: u64) -> Result<String, LlmError> {
        let client = reqwest::Client::new();
        let max_attempts = 60;

        for attempt in 0..max_attempts {
            tokio::time::sleep(tokio::time::Duration::from_secs(interval)).await;

            let resp = client
                .post(GITHUB_ACCESS_TOKEN_URL)
                .header("Accept", "application/json")
                .header("User-Agent", "GithubCopilot/1.155.0")
                .json(&serde_json::json!({
                    "client_id": GITHUB_CLIENT_ID,
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code"
                }))
                .send()
                .await
                .map_err(|e| LlmError::HttpError(format!("Failed to poll for token: {}", e)))?;

            if !resp.status().is_success() {
                continue;
            }

            let token_resp = resp
                .json::<AccessTokenResponse>()
                .await
                .map_err(|e| LlmError::HttpError(e.to_string()))?;

            if let Some(token) = token_resp.access_token {
                eprintln!("✅ Authentication successful!");
                return Ok(token);
            }

            if let Some(error) = token_resp.error {
                if error != "authorization_pending" {
                    return Err(LlmError::AuthError);
                }
            }

            if attempt % 6 == 0 {
                eprintln!(
                    "⏳ Waiting for authorization... (attempt {}/{})",
                    attempt + 1,
                    max_attempts
                );
            }
        }

        Err(LlmError::AuthError)
    }

    async fn refresh_api_key(&mut self) -> Result<(), LlmError> {
        let client = reqwest::Client::new();

        let resp = client
            .get(GITHUB_COPILOT_TOKEN_URL)
            .header("Authorization", format!("token {}", self.access_token))
            .header("Accept", "application/json")
            .header("User-Agent", "GithubCopilot/1.155.0")
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("Failed to refresh API key: {}", e)))?;

        if !resp.status().is_success() {
            if resp.status().as_u16() == 401 {
                let token_dir = Self::get_token_dir();
                self.access_token = Self::perform_oauth_flow(&token_dir).await?;
                return Box::pin(self.refresh_api_key()).await;
            }

            return Err(LlmError::ApiError {
                status: resp.status().as_u16(),
                message: resp.text().await.unwrap_or_default(),
            });
        }

        let api_key_info = resp
            .json::<CopilotApiKeyResponse>()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        self.api_key = api_key_info.token.clone();

        let token_dir = Self::get_token_dir();
        let api_key_path = token_dir.join("api-key.json");
        std::fs::write(&api_key_path, serde_json::to_string(&api_key_info).unwrap())
            .map_err(|e| LlmError::ProviderError(format!("Failed to cache API key: {}", e)))?;

        Ok(())
    }

    pub fn with_editor_version(mut self, version: impl Into<String>) -> Self {
        self.editor_version = version.into();
        self
    }

    pub fn with_integration_id(mut self, id: impl Into<String>) -> Self {
        self.integration_id = id.into();
        self
    }

    fn convert_request(&self, request: ChatCompletionRequest) -> CopilotChatRequest {
        let messages: Vec<CopilotMessage> = request
            .messages
            .into_iter()
            .map(|msg| match msg {
                Message::System { content, .. } => CopilotMessage {
                    role: "system".to_string(),
                    content,
                },
                Message::User { content, .. } => CopilotMessage {
                    role: "user".to_string(),
                    content,
                },
                Message::Assistant { content, .. } => CopilotMessage {
                    role: "assistant".to_string(),
                    content: content.unwrap_or_default(),
                },
                Message::Tool { content, .. } => CopilotMessage {
                    role: "tool".to_string(),
                    content,
                },
            })
            .collect();

        CopilotChatRequest {
            messages,
            model: request.model,
            temperature: request.temperature,
            top_p: request.top_p,
            stream: request.stream,
            stop: request.stop,
            max_tokens: request.max_tokens,
        }
    }

    fn convert_response(&self, response: CopilotChatResponse) -> ChatCompletionResponse {
        ChatCompletionResponse {
            id: response.id,
            object: response.object,
            created: response.created,
            model: response.model,
            choices: response
                .choices
                .into_iter()
                .map(|c| Choice {
                    index: c.index as u32,
                    message: Message::Assistant {
                        content: Some(c.message.content),
                        refusal: None,
                        tool_calls: None,
                    },
                    finish_reason: c.finish_reason,
                })
                .collect(),
            usage: response.usage.map(|u| Usage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
            }),
        }
    }

    async fn make_request(
        &mut self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        if self.api_key.is_empty() {
            self.refresh_api_key().await?;
        }

        let url = format!("{}/chat/completions", GITHUB_COPILOT_API_BASE);
        let copilot_req = self.convert_request(request.clone());

        let initiator = if request
            .messages
            .iter()
            .any(|m| matches!(m, Message::Assistant { .. } | Message::Tool { .. }))
        {
            "agent"
        } else {
            "user"
        };

        let request_id = uuid::Uuid::new_v4().to_string();
        let resp = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .header("editor-version", "vscode/1.95.0")
            .header("editor-plugin-version", "copilot-chat/0.26.7")
            .header("Copilot-Integration-Id", "vscode-chat")
            .header("User-Agent", "GitHubCopilotChat/0.26.7")
            .header("openai-intent", "conversation-panel")
            .header("x-github-api-version", "2025-04-01")
            .header("x-request-id", &request_id)
            .header("x-vscode-user-agent-library-version", "electron-fetch")
            .header("X-Initiator", initiator)
            .json(&copilot_req)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if resp.status().as_u16() == 401 {
            self.refresh_api_key().await?;
            return Box::pin(async move { self.make_request(request).await }).await;
        }

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let text = resp.text().await.unwrap_or_default();

            if status == 429 {
                return Err(LlmError::RateLimitError);
            }

            return Err(LlmError::ApiError {
                status,
                message: text,
            });
        }

        let text = resp
            .text()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;
        let copilot_resp: CopilotChatResponse = serde_json::from_str(&text)
            .map_err(|e| LlmError::HttpError(format!("error decoding response body: {}", e)))?;

        Ok(self.convert_response(copilot_resp))
    }

    async fn make_embedding_request(
        &mut self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LlmError> {
        if self.api_key.is_empty() {
            self.refresh_api_key().await?;
        }

        let url = format!("{}/embeddings", GITHUB_COPILOT_API_BASE);
        let copilot_req = CopilotEmbeddingRequest {
            model: request.model.clone(),
            input: vec![request.input.clone()],
        };

        let request_id = uuid::Uuid::new_v4().to_string();
        let resp = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .header("editor-version", "vscode/1.95.0")
            .header("editor-plugin-version", "copilot-chat/0.26.7")
            .header("Copilot-Integration-Id", "vscode-chat")
            .header("User-Agent", "GitHubCopilotChat/0.26.7")
            .header("openai-intent", "conversation-panel")
            .header("x-github-api-version", "2025-04-01")
            .header("x-request-id", &request_id)
            .header("x-vscode-user-agent-library-version", "electron-fetch")
            .header("X-Initiator", "user")
            .json(&copilot_req)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if resp.status().as_u16() == 401 {
            self.refresh_api_key().await?;
            return Box::pin(async move { self.make_embedding_request(request).await }).await;
        }

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let text = resp.text().await.unwrap_or_default();

            if status == 429 {
                return Err(LlmError::RateLimitError);
            }

            return Err(LlmError::ApiError {
                status,
                message: text,
            });
        }

        let text = resp
            .text()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;
        let copilot_resp: CopilotEmbeddingResponse = serde_json::from_str(&text)
            .map_err(|e| LlmError::HttpError(format!("error decoding response body: {}", e)))?;

        Ok(EmbeddingResponse {
            id: uuid::Uuid::new_v4().to_string(),
            object: if copilot_resp.object.is_empty() {
                "list".to_string()
            } else {
                copilot_resp.object
            },
            data: copilot_resp
                .data
                .into_iter()
                .map(|d| llmg_core::types::Embedding {
                    index: d.index,
                    object: d.object,
                    embedding: d.embedding,
                })
                .collect(),
            model: copilot_resp.model,
            usage: Usage {
                prompt_tokens: copilot_resp.usage.prompt_tokens,
                completion_tokens: copilot_resp.usage.completion_tokens,
                total_tokens: copilot_resp.usage.total_tokens,
            },
        })
    }

    pub fn get_models() -> Vec<String> {
        vec![
            "gpt-4".to_string(),
            "gpt-4o".to_string(),
            "gpt-4o-mini".to_string(),
            "gpt-3.5-turbo".to_string(),
            "o1-preview".to_string(),
            "o1-mini".to_string(),
            "claude-3-5-sonnet".to_string(),
            "text-embedding-3-small".to_string(),
        ]
    }
}

#[async_trait::async_trait]
impl Provider for GitHubCopilotClient {
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        let mut client = self.clone();
        client.make_request(request).await
    }

    async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        let mut client = self.clone();
        client.make_embedding_request(request).await
    }
    fn provider_name(&self) -> &'static str {
        "github_copilot"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_copilot_client_with_api_key() {
        let client = GitHubCopilotClient::with_api_key("test-api-key", "test-access-token");
        assert_eq!(client.provider_name(), "github_copilot");
    }

    #[test]
    fn test_request_conversion() {
        let client = GitHubCopilotClient::with_api_key("test-key", "test-token");

        let request = ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![
                Message::System {
                    content: "You are a helpful coding assistant".to_string(),
                    name: None,
                },
                Message::User {
                    content: "Write a Python function".to_string(),
                    name: None,
                },
            ],
            temperature: Some(0.7),
            max_tokens: Some(1000),
            stream: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            user: None,
            tools: None,
            tool_choice: None,
        };

        let copilot_req = client.convert_request(request);

        assert_eq!(copilot_req.model, "gpt-4");
        assert_eq!(copilot_req.messages.len(), 2);
        assert_eq!(copilot_req.messages[0].role, "system");
        assert_eq!(copilot_req.messages[1].role, "user");
    }
}
