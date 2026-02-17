use llmg_core::types::{ChatCompletionRequest, ChatCompletionResponse};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// A simple in-memory cache for LLM responses
pub struct LlmCache {
    entries: Arc<RwLock<HashMap<String, CacheEntry>>>,
}

struct CacheEntry {
    response: ChatCompletionResponse,
    expires_at: std::time::Instant,
}

impl LlmCache {
    pub fn new() -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Generate a cache key from a request
    fn make_key(request: &ChatCompletionRequest) -> String {
        // Simple hash of model and messages
        let json = serde_json::to_string(request).unwrap_or_default();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        use std::hash::{Hash, Hasher};
        json.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    pub async fn get(&self, request: &ChatCompletionRequest) -> Option<ChatCompletionResponse> {
        let key = Self::make_key(request);
        let entries = self.entries.read().await;

        if let Some(entry) = entries.get(&key) {
            if entry.expires_at > std::time::Instant::now() {
                return Some(entry.response.clone());
            }
        }
        None
    }

    pub async fn set(
        &self,
        request: &ChatCompletionRequest,
        response: ChatCompletionResponse,
        ttl_secs: u64,
    ) {
        let key = Self::make_key(request);
        let mut entries = self.entries.write().await;

        entries.insert(
            key,
            CacheEntry {
                response,
                expires_at: std::time::Instant::now() + std::time::Duration::from_secs(ttl_secs),
            },
        );
    }

    pub async fn clear_expired(&self) {
        let mut entries = self.entries.write().await;
        let now = std::time::Instant::now();
        entries.retain(|_, v| v.expires_at > now);
    }
}

impl Default for LlmCache {
    fn default() -> Self {
        Self::new()
    }
}
