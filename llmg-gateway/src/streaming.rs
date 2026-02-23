//! Server-Sent Events (SSE) streaming support for LLMG

use axum::response::{IntoResponse, Response};
use futures::stream::{self, Stream, StreamExt};
use llmg_core::provider::ChatCompletionStream;
use llmg_core::streaming::{ChatCompletionChunk, ChoiceDelta, DeltaContent};
use std::pin::Pin;
use std::task::{Context, Poll};

type SseStreamInner = Pin<
    Box<
        dyn Stream<Item = Result<ChatCompletionChunk, Box<dyn std::error::Error + Send + Sync>>>
            + Send,
    >,
>;

/// SSE stream wrapper
pub struct SseStream {
    inner: SseStreamInner,
}

impl SseStream {
    /// Create a new SSE stream from a provider stream
    #[allow(dead_code)]
    pub fn new<S>(stream: S) -> Self
    where
        S: Stream<Item = Result<ChatCompletionChunk, Box<dyn std::error::Error + Send + Sync>>>
            + Send
            + 'static,
    {
        Self {
            inner: Box::pin(stream),
        }
    }

    /// Create an SSE stream from a provider's ChatCompletionStream
    /// Converts LlmError to boxed error for compatibility
    pub fn from_provider_stream(stream: ChatCompletionStream) -> Self {
        let mapped = stream.map(|result| {
            result.map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })
        });
        Self {
            inner: Box::pin(mapped),
        }
    }

    /// Create a simple text stream
    #[allow(dead_code)]
    pub fn from_text(model: String, text: String) -> Self {
        let chunks: Vec<ChatCompletionChunk> = text
            .chars()
            .collect::<Vec<_>>()
            .chunks(4)
            .enumerate()
            .map(|(i, chars)| {
                let content: String = chars.iter().collect();
                ChatCompletionChunk {
                    id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                    object: "chat.completion.chunk".to_string(),
                    created: chrono::Utc::now().timestamp(),
                    model: model.clone(),
                    choices: vec![ChoiceDelta {
                        index: 0,
                        delta: DeltaContent {
                            role: if i == 0 {
                                Some("assistant".to_string())
                            } else {
                                None
                            },
                            content: Some(content),
                            tool_calls: None,
                        },
                        finish_reason: None,
                    }],
                }
            })
            .collect();

        let stream = stream::iter(chunks)
            .map(Ok::<_, Box<dyn std::error::Error + Send + Sync>>)
            .chain(stream::once(async move {
                Ok(ChatCompletionChunk {
                    id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                    object: "chat.completion.chunk".to_string(),
                    created: chrono::Utc::now().timestamp(),
                    model: model.clone(),
                    choices: vec![ChoiceDelta {
                        index: 0,
                        delta: DeltaContent::default(),
                        finish_reason: Some("stop".to_string()),
                    }],
                })
            }));

        Self {
            inner: Box::pin(stream),
        }
    }
}

impl Stream for SseStream {
    type Item = Result<ChatCompletionChunk, Box<dyn std::error::Error + Send + Sync>>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.poll_next_unpin(cx)
    }
}

impl IntoResponse for SseStream {
    fn into_response(self) -> Response {
        use axum::response::sse::{Event, Sse};

        let stream = self.inner.map(|result| {
            result.map(|chunk| {
                let data = serde_json::to_string(&chunk).unwrap_or_default();
                Event::default().data(data)
            })
        });

        Sse::new(stream)
            .keep_alive(axum::response::sse::KeepAlive::default().text("\n"))
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse_stream_creation() {
        let _stream = SseStream::from_text("gpt-4".to_string(), "Hello".to_string());
    }

    #[test]
    fn test_chat_completion_chunk_serialization() {
        let chunk = ChatCompletionChunk {
            id: "test-id".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1234567890,
            model: "gpt-4".to_string(),
            choices: vec![ChoiceDelta {
                index: 0,
                delta: DeltaContent {
                    role: Some("assistant".to_string()),
                    content: Some("Hello".to_string()),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
        };

        let json = serde_json::to_string(&chunk).unwrap();
        assert!(json.contains("chat.completion.chunk"));
    }
}
