//! Google Antigravity API client for LLMG
//!
//! Antigravity uses Google OAuth with PKCE (Proof Key for Code Exchange).
//!
//! Authentication Flow:
//! 1. Generate PKCE code verifier and challenge
//! 2. Open browser for user to authorize
//! 3. Exchange authorization code for tokens
//! 4. Call loadCodeAssist to get project ID
//! 5. Store refresh_token|project_id for future use
//!
//! Environment variables:
//! - ANTIGRAVITY_REFRESH_TOKEN: Cached refresh token with project ID (format: "token|project_id")
//! - ANTIGRAVITY_ACCESS_TOKEN: Cached access token
//! - ANTIGRAVITY_TOKEN_DIR: Directory to store tokens (default: ~/.config/llmg/antigravity)

use llmg_core::{
    provider::{LlmError, Provider},
    types::{
        ChatCompletionRequest, ChatCompletionResponse, Choice, Embedding, EmbeddingRequest,
        EmbeddingResponse, Message, Usage,
    },
};
// use serde::Serialize; // removed unused import
use std::path::PathBuf;

/// Google OAuth constants for Antigravity
const ANTIGRAVITY_CLIENT_ID: &str =
    "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com";
const ANTIGRAVITY_CLIENT_SECRET: &str = "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";
const GOOGLE_AUTH_URL: &str = "https://accounts.google.com/o/oauth2/v2/auth";
const GOOGLE_TOKEN_URL: &str = "https://oauth2.googleapis.com/token";

/// Antigravity endpoints
const ANTIGRAVITY_ENDPOINT_DAILY: &str = "https://daily-cloudcode-pa.sandbox.googleapis.com";
const ANTIGRAVITY_ENDPOINT_AUTOPUSH: &str = "https://autopush-cloudcode-pa.sandbox.googleapis.com";
const ANTIGRAVITY_ENDPOINT_PROD: &str = "https://cloudcode-pa.googleapis.com";

/// Default redirect URI for CLI applications
const DEFAULT_REDIRECT_URI: &str = "http://localhost:51121/oauth-callback";

/// Google Antigravity API client
#[derive(Debug, Clone)]
pub struct AntigravityClient {
    http_client: reqwest::Client,
    access_token: String,
    refresh_token: String,
    project_id: String,
    endpoint: String,
    api_version: String,
}

/// PKCE code pair
#[derive(Debug)]
struct PkcePair {
    verifier: String,
    challenge: String,
}

#[derive(Debug, serde::Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: i32,
    token_type: String,
}

#[derive(Debug, serde::Deserialize)]
struct LoadCodeAssistResponse {
    #[serde(rename = "cloudaicompanionProject")]
    cloudaicompanion_project: Option<serde_json::Value>,
}

#[derive(Debug, serde::Serialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
}

#[derive(Debug, serde::Serialize)]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

#[derive(Debug, serde::Serialize)]
struct GeminiPart {
    text: String,
}

#[derive(Debug, serde::Serialize)]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
}

#[derive(Debug, serde::Deserialize)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Debug, serde::Deserialize)]
struct GeminiCandidate {
    content: GeminiContentResponse,
    #[serde(rename = "finishReason")]
    finish_reason: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct GeminiContentResponse {
    parts: Vec<GeminiPartResponse>,
}

#[derive(Debug, serde::Deserialize)]
struct GeminiPartResponse {
    text: Option<String>,
}

/// Gemini embedding response
#[derive(Debug, serde::Deserialize)]
struct GeminiEmbeddingResponse {
    embedding: GeminiEmbeddingValues,
}

#[derive(Debug, serde::Deserialize)]
struct GeminiEmbeddingValues {
    values: Vec<f32>,
}

#[derive(Debug, serde::Deserialize)]
struct GeminiUsageMetadata {
    #[serde(rename = "promptTokenCount")]
    prompt_token_count: Option<i32>,
    #[serde(rename = "candidatesTokenCount")]
    candidates_token_count: Option<i32>,
    #[serde(rename = "totalTokenCount")]
    total_token_count: Option<i32>,
}

impl AntigravityClient {
    /// Create a new Antigravity client with OAuth flow
    pub async fn new() -> Result<Self, LlmError> {
        let token_dir = Self::get_token_dir();
        std::fs::create_dir_all(&token_dir).map_err(|e| {
            LlmError::ProviderError(format!("Failed to create token directory: {}", e))
        })?;

        let (access_token, refresh_token, project_id) =
            Self::load_or_refresh_tokens(&token_dir).await?;

        Ok(Self {
            http_client: reqwest::Client::new(),
            access_token,
            refresh_token,
            project_id,
            endpoint: ANTIGRAVITY_ENDPOINT_DAILY.to_string(),
            api_version: "v1".to_string(),
        })
    }

    /// Create with explicit tokens (for testing or pre-authenticated scenarios)
    pub fn with_tokens(
        access_token: impl Into<String>,
        refresh_token: impl Into<String>,
        project_id: impl Into<String>,
    ) -> Self {
        Self {
            http_client: reqwest::Client::new(),
            access_token: access_token.into(),
            refresh_token: refresh_token.into(),
            project_id: project_id.into(),
            endpoint: ANTIGRAVITY_ENDPOINT_DAILY.to_string(),
            api_version: "v1".to_string(),
        }
    }

    fn get_token_dir() -> PathBuf {
        std::env::var("ANTIGRAVITY_TOKEN_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                dirs::config_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join("llmg/antigravity")
            })
    }

    async fn load_or_refresh_tokens(
        token_dir: &std::path::Path,
    ) -> Result<(String, String, String), LlmError> {
        let refresh_token_path = token_dir.join("refresh-token");

        if let Ok(content) = std::fs::read_to_string(&refresh_token_path) {
            let parts: Vec<&str> = content.trim().split('|').collect();
            if parts.len() >= 2 {
                let refresh_token = parts[0].to_string();
                let project_id = parts[1].to_string();

                // Try to refresh the access token
                match Self::refresh_access_token(&refresh_token).await {
                    Ok(access_token) => return Ok((access_token, refresh_token, project_id)),
                    Err(_) => {
                        eprintln!("Failed to refresh token, initiating new OAuth flow...");
                    }
                }
            }
        }

        // No valid cached tokens, initiate OAuth flow
        Self::perform_oauth_flow(token_dir).await
    }

    async fn perform_oauth_flow(
        _token_dir: &std::path::Path,
    ) -> Result<(String, String, String), LlmError> {
        // Generate PKCE pair
        let pkce = Self::generate_pkce()?;

        // Build authorization URL
        let auth_url = format!(
            "{}?client_id={}&response_type=code&redirect_uri={}&scope={}&code_challenge={}&code_challenge_method=S256&access_type=offline&prompt=consent",
            GOOGLE_AUTH_URL,
            ANTIGRAVITY_CLIENT_ID,
            urlencoding::encode(DEFAULT_REDIRECT_URI),
            urlencoding::encode("https://www.googleapis.com/auth/cloud-platform https://www.googleapis.com/auth/userinfo.email"),
            pkce.challenge
        );

        eprintln!("\n🔐 Google Antigravity Authentication Required");
        eprintln!("Please visit the following URL to authorize:");
        eprintln!("{}\n", auth_url);
        eprintln!("Note: Since this is a CLI application, you'll need to manually copy the 'code' parameter from the redirect URL after authorization.");
        eprintln!("The redirect will go to: {}\n", DEFAULT_REDIRECT_URI);

        // For CLI applications, we need to prompt for the code
        // In a real implementation, you'd start a local server to capture the callback
        eprintln!("Please enter the authorization code from the redirect URL:");

        // For now, return an error with instructions
        Err(LlmError::AuthError)
    }

    fn generate_pkce() -> Result<PkcePair, LlmError> {
        use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
        use rand::Rng;
        use sha2::{Digest, Sha256};

        // Generate random code verifier (43-128 chars)
        let mut rng = rand::thread_rng();
        let verifier_bytes: Vec<u8> = (0..64).map(|_| rng.gen::<u8>()).collect();
        let verifier = URL_SAFE_NO_PAD.encode(&verifier_bytes);

        // Create code challenge (SHA256 hash of verifier)
        let mut hasher = Sha256::new();
        hasher.update(verifier.as_bytes());
        let challenge = URL_SAFE_NO_PAD.encode(hasher.finalize());

        Ok(PkcePair {
            verifier,
            challenge,
        })
    }

    async fn exchange_code_for_tokens(
        code: &str,
        pkce_verifier: &str,
    ) -> Result<(String, String), LlmError> {
        let client = reqwest::Client::new();

        let params = [
            ("client_id", ANTIGRAVITY_CLIENT_ID),
            ("client_secret", ANTIGRAVITY_CLIENT_SECRET),
            ("code", code),
            ("grant_type", "authorization_code"),
            ("redirect_uri", DEFAULT_REDIRECT_URI),
            ("code_verifier", pkce_verifier),
        ];

        let resp = client
            .post(GOOGLE_TOKEN_URL)
            .form(&params)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("Failed to exchange code: {}", e)))?;

        if !resp.status().is_success() {
            return Err(LlmError::ApiError {
                status: resp.status().as_u16(),
                message: resp.text().await.unwrap_or_default(),
            });
        }

        let token_resp = resp
            .json::<TokenResponse>()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        let refresh_token = token_resp.refresh_token.ok_or(LlmError::AuthError)?;

        Ok((token_resp.access_token, refresh_token))
    }

    async fn refresh_access_token(refresh_token: &str) -> Result<String, LlmError> {
        let client = reqwest::Client::new();

        let params = [
            ("client_id", ANTIGRAVITY_CLIENT_ID),
            ("client_secret", ANTIGRAVITY_CLIENT_SECRET),
            ("refresh_token", refresh_token),
            ("grant_type", "refresh_token"),
        ];

        let resp = client
            .post(GOOGLE_TOKEN_URL)
            .form(&params)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("Failed to refresh token: {}", e)))?;

        if !resp.status().is_success() {
            return Err(LlmError::ApiError {
                status: resp.status().as_u16(),
                message: resp.text().await.unwrap_or_default(),
            });
        }

        let token_resp = resp
            .json::<TokenResponse>()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        Ok(token_resp.access_token)
    }

    async fn fetch_project_id(&self) -> Result<String, LlmError> {
        let client = reqwest::Client::new();

        let url = format!("{}/v1internal:loadCodeAssist", self.endpoint);

        let resp = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.access_token))
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "metadata": {
                    "ideType": "IDE_UNSPECIFIED",
                    "platform": "PLATFORM_UNSPECIFIED",
                    "pluginType": "GEMINI"
                }
            }))
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("Failed to fetch project ID: {}", e)))?;

        if !resp.status().is_success() {
            return Err(LlmError::ApiError {
                status: resp.status().as_u16(),
                message: resp.text().await.unwrap_or_default(),
            });
        }

        let load_resp = resp
            .json::<LoadCodeAssistResponse>()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        // Extract project ID from response
        if let Some(project) = load_resp.cloudaicompanion_project {
            if let Some(id) = project.as_str() {
                return Ok(id.to_string());
            }
            if let Some(obj) = project.as_object() {
                if let Some(id) = obj.get("id").and_then(|v| v.as_str()) {
                    return Ok(id.to_string());
                }
            }
        }

        // Default project ID
        Ok("rising-fact-p41fc".to_string())
    }

    /// Convert OpenAI format to Gemini format
    fn convert_request(&self, request: ChatCompletionRequest) -> GeminiRequest {
        let mut contents = Vec::new();
        let mut system_instruction = None;

        for msg in request.messages {
            match msg {
                Message::System { content, .. } => {
                    system_instruction = Some(GeminiContent {
                        role: "user".to_string(),
                        parts: vec![GeminiPart { text: content }],
                    });
                }
                Message::User { content, .. } => {
                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts: vec![GeminiPart { text: content }],
                    });
                }
                Message::Assistant { content, .. } => {
                    if let Some(content) = content {
                        contents.push(GeminiContent {
                            role: "model".to_string(),
                            parts: vec![GeminiPart { text: content }],
                        });
                    }
                }
                Message::Tool { content, .. } => {
                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts: vec![GeminiPart { text: content }],
                    });
                }
            }
        }

        let generation_config = GeminiGenerationConfig {
            temperature: request.temperature,
            top_p: request.top_p,
            max_output_tokens: request.max_tokens.map(|t| t as i32),
            stop_sequences: request.stop,
        };

        GeminiRequest {
            contents,
            system_instruction,
            generation_config: Some(generation_config),
        }
    }

    fn convert_response(&self, response: GeminiResponse, model: &str) -> ChatCompletionResponse {
        let content = response
            .candidates
            .first()
            .and_then(|c| c.content.parts.first())
            .and_then(|p| p.text.clone())
            .unwrap_or_default();

        let finish_reason = response
            .candidates
            .first()
            .and_then(|c| c.finish_reason.clone())
            .map(|fr| match fr.as_str() {
                "STOP" => "stop".to_string(),
                "MAX_TOKENS" => "length".to_string(),
                "SAFETY" => "content_filter".to_string(),
                _ => "stop".to_string(),
            });

        let usage = response.usage_metadata.map(|u| Usage {
            prompt_tokens: u.prompt_token_count.unwrap_or(0) as u32,
            completion_tokens: u.candidates_token_count.unwrap_or(0) as u32,
            total_tokens: u.total_token_count.unwrap_or(0) as u32,
        });

        ChatCompletionResponse {
            id: format!("ag-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: model.to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message::Assistant {
                    content: Some(content),
                    refusal: None,
                    tool_calls: None,
                },
                finish_reason,
            }],
            usage,
        }
    }

    fn map_model_name(&self, model: &str) -> String {
        let model = if model.contains('/') {
            model.split('/').next_back().unwrap_or(model)
        } else {
            model
        };

        match model {
            "gpt-4" | "gpt-4o" => "gemini-2.0-pro-exp-02-05",
            "gpt-4o-mini" => "gemini-2.0-flash",
            "gpt-3.5-turbo" => "gemini-1.5-flash",
            m if m.starts_with("gemini-") => m,
            _ => "gemini-2.0-flash",
        }
        .to_string()
    }

    fn build_url(&self, model: &str) -> String {
        let mapped_model = self.map_model_name(model);
        format!(
            "{}/{}/projects/{}/locations/us-central1/publishers/google/models/{}:generateContent",
            self.endpoint, self.api_version, self.project_id, mapped_model
        )
    }

    async fn make_request(
        &mut self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        let model = request.model.clone();
        let url = self.build_url(&model);
        let gemini_req = self.convert_request(request.clone());

        let resp = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.access_token))
            .header("Content-Type", "application/json")
            .json(&gemini_req)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if resp.status().as_u16() == 401 {
            // Token expired, try to refresh
            self.access_token = Self::refresh_access_token(&self.refresh_token).await?;
            return Box::pin(async move { self.make_request(request).await }).await;
        }

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let text = resp.text().await.unwrap_or_default();
            return Err(LlmError::ApiError {
                status,
                message: text,
            });
        }

        let gemini_resp: GeminiResponse = resp
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        Ok(self.convert_response(gemini_resp, &model))
    }

    pub fn get_models() -> Vec<String> {
        vec![
            "gemini-2.0-pro-exp-02-05".to_string(),
            "gemini-2.0-flash".to_string(),
            "gemini-2.0-flash-lite".to_string(),
            "gemini-1.5-pro".to_string(),
            "gemini-1.5-flash".to_string(),
        ]
    }
}

#[async_trait::async_trait]
impl Provider for AntigravityClient {
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        let mut client = self.clone();
        client.make_request(request).await
    }

    async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        let model_name = self.map_model_name(&request.model);
        let url = format!(
            "{}/{}/projects/{}/locations/us-central1/publishers/google/models/{}:predict",
            self.endpoint, self.api_version, self.project_id, model_name
        );

        let body = serde_json::json!({
            "instances": [{"content": request.input}]
        });

        let resp = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.access_token))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let text = resp.text().await.unwrap_or_default();
            return Err(LlmError::ApiError {
                status,
                message: text,
            });
        }

        let embed_resp: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        // Response: {"predictions": [{"embeddings": {"values": [...]}}]}
        let values = embed_resp["predictions"][0]["embeddings"]["values"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect::<Vec<f32>>()
            })
            .unwrap_or_default();

        Ok(EmbeddingResponse {
            id: format!("ag-emb-{}", uuid::Uuid::new_v4()),
            object: "list".to_string(),
            data: vec![Embedding {
                index: 0,
                object: "embedding".to_string(),
                embedding: values,
            }],
            model: model_name,
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
        })
    }
    fn provider_name(&self) -> &'static str {
        "antigravity"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_antigravity_with_tokens() {
        let client = AntigravityClient::with_tokens("access", "refresh", "project-123");
        assert_eq!(client.provider_name(), "antigravity");
    }

    #[test]
    fn test_model_name_mapping() {
        let client = AntigravityClient::with_tokens("a", "r", "p");
        assert_eq!(client.map_model_name("gpt-4"), "gemini-2.0-pro-exp-02-05");
        assert_eq!(client.map_model_name("gemini-1.5-pro"), "gemini-1.5-pro");
    }
}

