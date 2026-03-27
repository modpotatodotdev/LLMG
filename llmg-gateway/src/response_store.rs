//! Response store for OpenAI Responses API
//!
//! Provides persistent storage for responses, enabling retrieval, deletion,
//! cancellation, and conversation continuation via `previous_response_id`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// A stored response entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredResponse {
    pub id: String,
    pub object: String,
    pub status: String,
    pub created_at: u64,
    pub completed_at: Option<u64>,
    pub model: String,
    pub output: Vec<serde_json::Value>,
    pub error: Option<StoredError>,
    pub incomplete_details: Option<StoredIncompleteDetails>,
    pub instructions: Option<String>,
    pub max_output_tokens: Option<u32>,
    pub previous_response_id: Option<String>,
    pub reasoning: Option<StoredReasoning>,
    pub store: bool,
    pub service_tier: Option<String>,
    pub temperature: Option<f32>,
    pub text: StoredTextFormat,
    pub tool_choice: Option<serde_json::Value>,
    pub tools: Option<Vec<serde_json::Value>>,
    pub top_p: Option<f32>,
    pub truncation: Option<String>,
    pub usage: Option<StoredUsage>,
    pub metadata: Option<serde_json::Value>,
    pub user: Option<String>,
    pub parallel_tool_calls: Option<bool>,
    /// Input items that produced this response (for input_items endpoint)
    pub input_items: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredError {
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredIncompleteDetails {
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredReasoning {
    pub effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredTextFormat {
    pub format: StoredTextFormatType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StoredTextFormatType {
    Text,
    JsonObject,
    #[serde(rename = "json_schema")]
    JsonSchema {
        name: String,
        schema: serde_json::Value,
    },
}

impl Default for StoredTextFormat {
    fn default() -> Self {
        Self {
            format: StoredTextFormatType::Text,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredUsage {
    pub input_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<StoredInputTokensDetails>,
    pub output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<StoredOutputTokensDetails>,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredInputTokensDetails {
    pub cached_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredOutputTokensDetails {
    pub reasoning_tokens: u32,
}

/// Thread-safe response store
pub struct ResponseStore {
    entries: Arc<RwLock<HashMap<String, StoredResponse>>>,
}

impl Clone for ResponseStore {
    fn clone(&self) -> Self {
        Self {
            entries: Arc::clone(&self.entries),
        }
    }
}

impl ResponseStore {
    pub fn new() -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Store a response
    pub async fn store(&self, response: StoredResponse) {
        let mut entries = self.entries.write().await;
        entries.insert(response.id.clone(), response);
    }

    /// Retrieve a response by ID
    pub async fn get(&self, id: &str) -> Option<StoredResponse> {
        let entries = self.entries.read().await;
        entries.get(id).cloned()
    }

    /// Delete a response by ID
    pub async fn delete(&self, id: &str) -> bool {
        let mut entries = self.entries.write().await;
        entries.remove(id).is_some()
    }

    /// Update response status (e.g., mark as cancelled)
    pub async fn update_status(&self, id: &str, status: &str) -> bool {
        let mut entries = self.entries.write().await;
        if let Some(entry) = entries.get_mut(id) {
            entry.status = status.to_string();
            true
        } else {
            false
        }
    }

    /// List input items for a response
    pub async fn get_input_items(&self, id: &str) -> Option<Vec<serde_json::Value>> {
        let entries = self.entries.read().await;
        entries.get(id).map(|r| r.input_items.clone())
    }

    /// Check if response exists
    pub async fn exists(&self, id: &str) -> bool {
        let entries = self.entries.read().await;
        entries.contains_key(id)
    }
}

impl Default for ResponseStore {
    fn default() -> Self {
        Self::new()
    }
}
