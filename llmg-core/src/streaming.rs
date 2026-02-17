//! Streaming types for chat completions
//!
//! Provides SSE-compatible chunk types for streaming responses.

use serde::{Deserialize, Serialize};

/// SSE event for streaming chat completions (OpenAI-compatible format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    /// Unique identifier for the chunk
    pub id: String,
    /// Object type (always "chat.completion.chunk")
    pub object: String,
    /// Unix timestamp
    pub created: i64,
    /// Model used
    pub model: String,
    /// Choice deltas
    pub choices: Vec<ChoiceDelta>,
}

impl ChatCompletionChunk {
    /// Create a new chunk with a single choice
    pub fn new(
        id: String,
        model: String,
        index: u32,
        delta: DeltaContent,
        finish_reason: Option<String>,
    ) -> Self {
        Self {
            id,
            object: "chat.completion.chunk".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            choices: vec![ChoiceDelta {
                index,
                delta,
                finish_reason,
            }],
        }
    }

    /// Create a final chunk with finish_reason
    pub fn final_chunk(id: String, model: String, finish_reason: &str) -> Self {
        Self {
            id,
            object: "chat.completion.chunk".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            choices: vec![ChoiceDelta {
                index: 0,
                delta: DeltaContent::default(),
                finish_reason: Some(finish_reason.to_string()),
            }],
        }
    }

    /// Generate a new chunk ID
    pub fn generate_id() -> String {
        format!("chatcmpl-{}", uuid::Uuid::new_v4())
    }
}

/// Delta in a streaming choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceDelta {
    /// Index of the choice
    pub index: u32,
    /// Delta object
    pub delta: DeltaContent,
    /// Finish reason (if completed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Content delta for streaming responses
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeltaContent {
    /// Role (only sent in first chunk)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Content delta
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

impl DeltaContent {
    /// Create a role delta (for first chunk)
    pub fn role() -> Self {
        Self {
            role: Some("assistant".to_string()),
            content: None,
        }
    }

    /// Create a content delta
    pub fn content(text: impl Into<String>) -> Self {
        Self {
            role: None,
            content: Some(text.into()),
        }
    }

    /// Create an empty delta (for final chunk)
    pub fn empty() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_serialization() {
        let chunk = ChatCompletionChunk::new(
            "test-id".to_string(),
            "gpt-4".to_string(),
            0,
            DeltaContent::content("Hello"),
            None,
        );

        let json = serde_json::to_string(&chunk).unwrap();
        assert!(json.contains("chat.completion.chunk"));
        assert!(json.contains("Hello"));
    }

    #[test]
    fn test_final_chunk() {
        let chunk =
            ChatCompletionChunk::final_chunk("test-id".to_string(), "gpt-4".to_string(), "stop");

        assert_eq!(chunk.choices[0].finish_reason, Some("stop".to_string()));
        assert!(chunk.choices[0].delta.content.is_none());
    }
}
