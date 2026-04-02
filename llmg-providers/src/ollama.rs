#![allow(clippy::redundant_closure)]

use futures::{StreamExt, TryStreamExt};
use llmg_core::{
    provider::{ChatCompletionStream, LlmError, Provider},
    streaming::{ChatCompletionChunk, ChoiceDelta, DeltaContent},
    types::{
        ChatCompletionRequest, ChatCompletionResponse, Choice, EmbeddingRequest, EmbeddingResponse,
        Message, Usage,
    },
};
use std::future::Future;
use std::pin::Pin;
use tokio_util::codec::{FramedRead, LinesCodec};
use tokio_util::io::StreamReader;

/// Ollama API client
#[derive(Debug)]
pub struct OllamaClient {
    http_client: reqwest::Client,
    base_url: String,
}

/// Ollama chat request format
#[derive(Debug, serde::Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
}

#[derive(Debug, serde::Serialize, Default)]
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    repeat_penalty: Option<f32>,
}

#[derive(Debug, serde::Serialize)]
struct OllamaMessage {
    role: String,
    content: String,
}

/// Ollama chat response format
#[derive(Debug, serde::Deserialize)]
struct OllamaChatResponse {
    message: OllamaResponseMessage,
    #[serde(rename = "eval_count")]
    eval_count: Option<u32>,
    #[serde(rename = "prompt_eval_count")]
    prompt_eval_count: Option<u32>,
    done: bool,
}

#[derive(Debug, serde::Deserialize)]
struct OllamaStreamResponse {
    message: Option<OllamaResponseMessage>,
    #[serde(rename = "eval_count")]
    eval_count: Option<u32>,
    #[serde(rename = "prompt_eval_count")]
    prompt_eval_count: Option<u32>,
    done: bool,
}

#[derive(Debug, serde::Deserialize)]
struct OllamaResponseMessage {
    role: String,
    content: String,
}

impl OllamaClient {
    /// Create a new Ollama client with default localhost URL
    pub fn new() -> Self {
        Self {
            http_client: reqwest::Client::new(),
            base_url: "http://localhost:11434".to_string(),
        }
    }

    /// Create a new OllamaClient from environment variables.
    ///
    /// Reads `OLLAMA_BASE_URL` (default "http://localhost:11434").
    pub fn from_env() -> Self {
        let mut client = Self::new();
        if let Ok(base_url) = std::env::var("OLLAMA_BASE_URL") {
            client = client.with_base_url(base_url);
        }
        client
    }

    /// Create with custom base URL
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Convert OpenAI format to Ollama format
    fn convert_request(&self, request: ChatCompletionRequest) -> OllamaChatRequest {
        let messages: Vec<OllamaMessage> = request
            .messages
            .into_iter()
            .filter_map(|msg| match msg {
                Message::System { content, .. } => Some(OllamaMessage {
                    role: "system".to_string(),
                    content,
                }),
                Message::User { content, .. } => Some(OllamaMessage {
                    role: "user".to_string(),
                    content,
                }),
                Message::Assistant { content, .. } => content.map(|c| OllamaMessage {
                    role: "assistant".to_string(),
                    content: c,
                }),
                _ => None,
            })
            .collect();

        OllamaChatRequest {
            model: request.model,
            messages,
            stream: request.stream,
            options: Some(OllamaOptions {
                temperature: request.temperature,
                num_predict: request.max_tokens,
                top_p: request.top_p,
                stop: request.stop,
                // Ollama's repeat_penalty is similar to frequency_penalty
                // It penalizes tokens based on how frequently they appear
                repeat_penalty: request.frequency_penalty,
            }),
        }
    }

    /// Convert Ollama response to OpenAI format
    fn convert_response(
        &self,
        response: OllamaChatResponse,
        model: String,
    ) -> ChatCompletionResponse {
        let usage = if let (Some(prompt), Some(completion)) =
            (response.prompt_eval_count, response.eval_count)
        {
            Some(Usage {
                prompt_tokens: prompt,
                completion_tokens: completion,
                total_tokens: prompt + completion,
            })
        } else {
            None
        };

        ChatCompletionResponse {
            id: format!("ollama-{})", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            choices: vec![Choice {
                index: 0,
                message: Message::Assistant {
                    content: Some(response.message.content),
                    refusal: None,
                    tool_calls: None,
                },
                finish_reason: if response.done {
                    Some("stop".to_string())
                } else {
                    None
                },
            }],
            usage,
        }
    }

    async fn make_request(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        let model = request.model.clone();
        let ollama_req = self.convert_request(request);
        let url = format!("{}/api/chat", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .json(&ollama_req)
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

        let ollama_resp: OllamaChatResponse = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        Ok(self.convert_response(ollama_resp, model))
    }

    async fn make_stream_request(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionStream, LlmError> {
        let model = request.model.clone();
        let mut ollama_req = self.convert_request(request);
        ollama_req.stream = Some(true);

        let url = format!("{}/api/chat", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .json(&ollama_req)
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

        let chunk_id = ChatCompletionChunk::generate_id();
        let stream_model = model.clone();

        let byte_stream = response
            .bytes_stream()
            .map_err(|e| std::io::Error::other(e));

        let stream_reader = StreamReader::new(byte_stream);
        let lines_stream = FramedRead::new(stream_reader, LinesCodec::new());

        let stream = lines_stream
            .map_err(|e| LlmError::HttpError(e.to_string()))
            .then(move |line_res| {
                let chunk_id = chunk_id.clone();
                let stream_model = stream_model.clone();
                async move {
                    match line_res {
                        Ok(line) => parse_ollama_stream_line(&line, &chunk_id, &stream_model),
                        Err(e) => Err(LlmError::HttpError(e.to_string())),
                    }
                }
            })
            .try_filter_map(|chunk| async move { Ok(chunk) });

        Ok(Box::pin(stream) as ChatCompletionStream)
    }
}

impl Default for OllamaClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl Provider for OllamaClient {
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        self.make_request(request).await
    }

    fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ChatCompletionStream, LlmError>> + Send + '_>> {
        Box::pin(self.make_stream_request(request))
    }

    async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        let url = format!("{}/api/embeddings", self.base_url);

        let body = serde_json::json!({
            "model": request.model,
            "prompt": request.input,
        });

        let response = self
            .http_client
            .post(&url)
            .json(&body)
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

        let embedding_resp: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        let embedding: Vec<f32> = embedding_resp["embedding"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect()
            })
            .unwrap_or_default();

        Ok(EmbeddingResponse {
            id: format!("ollama-embed-{})", uuid::Uuid::new_v4()),
            object: "list".to_string(),
            data: vec![llmg_core::types::Embedding {
                index: 0,
                object: "embedding".to_string(),
                embedding,
            }],
            model: request.model,
            usage: llmg_core::types::Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
        })
    }
    fn provider_name(&self) -> &'static str {
        "ollama"
    }
}

fn parse_ollama_stream_line(
    line: &str,
    chunk_id: &str,
    model: &str,
) -> Result<Option<ChatCompletionChunk>, LlmError> {
    let line = line.trim();
    if line.is_empty() {
        return Ok(None);
    }

    let parsed: OllamaStreamResponse =
        serde_json::from_str(line).map_err(LlmError::SerializationError)?;

    if let Some(message) = parsed.message {
        if !message.content.is_empty() {
            return Ok(Some(ChatCompletionChunk {
                id: chunk_id.to_string(),
                object: "chat.completion.chunk".to_string(),
                created: chrono::Utc::now().timestamp(),
                model: model.to_string(),
                choices: vec![ChoiceDelta {
                    index: 0,
                    delta: DeltaContent::content(message.content),
                    finish_reason: None,
                }],
                usage: None,
            }));
        }
    }

    if parsed.done {
        return Ok(Some(ChatCompletionChunk::final_chunk(
            chunk_id.to_string(),
            model.to_string(),
            "stop",
        )));
    }

    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ollama_client_creation() {
        let client = OllamaClient::new();
        assert_eq!(client.provider_name(), "ollama");
        assert_eq!(client.base_url, "http://localhost:11434");
    }

    #[test]
    fn test_ollama_custom_url() {
        let client = OllamaClient::new().with_base_url("http://custom-server:8080");
        assert_eq!(client.base_url, "http://custom-server:8080");
    }

    #[test]
    fn test_request_conversion() {
        let client = OllamaClient::new();

        let request = ChatCompletionRequest {
            model: "llama2".to_string(),
            messages: vec![Message::User {
                content: "Hello!".to_string(),
                name: None,
            }],
            temperature: Some(0.7),
            max_tokens: Some(100),
            stream: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            user: None,
            tools: None,
            tool_choice: None,
            response_format: None,
        };

        let ollama_req = client.convert_request(request);

        assert_eq!(ollama_req.model, "llama2");
        assert_eq!(ollama_req.messages.len(), 1);
        assert_eq!(ollama_req.messages[0].role, "user");
        assert_eq!(ollama_req.messages[0].content, "Hello!");
    }
}

