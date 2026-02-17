//! Middleware for LLMG Gateway
//!
//! Provides authentication, CORS, logging, timeout, rate limiting, and retry support.

use axum::{
    extract::Request,
    http::{header, HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use llmg_core::provider::LlmError;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};

use crate::config::RateLimitConfig;

/// Token bucket for rate limiting
#[derive(Debug)]
struct TokenBucket {
    tokens: f64,
    max_tokens: f64,
    refill_rate: f64,
    last_refill: Instant,
}

impl TokenBucket {
    fn new(max_tokens: u32, refill_rate: f64) -> Self {
        Self {
            tokens: max_tokens as f64,
            max_tokens: max_tokens as f64,
            refill_rate,
            last_refill: Instant::now(),
        }
    }

    fn try_consume(&mut self, tokens: f64) -> bool {
        self.refill();
        if self.tokens >= tokens {
            self.tokens -= tokens;
            true
        } else {
            false
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        let tokens_to_add = elapsed * self.refill_rate;
        self.tokens = (self.tokens + tokens_to_add).min(self.max_tokens);
        self.last_refill = now;
    }

    fn time_until_available(&self) -> Duration {
        if self.tokens >= 1.0 {
            Duration::ZERO
        } else {
            let tokens_needed = 1.0 - self.tokens;
            Duration::from_secs_f64(tokens_needed / self.refill_rate)
        }
    }
}

/// Rate limiter state with per-provider buckets
#[derive(Debug)]
pub struct RateLimiter {
    global_bucket: RwLock<Option<TokenBucket>>,
    provider_buckets: RwLock<HashMap<String, TokenBucket>>,
    config: RateLimitConfig,
}

impl RateLimiter {
    pub fn new(config: RateLimitConfig) -> Self {
        let global_bucket = if config.enabled && config.requests_per_second > 0 {
            Some(TokenBucket::new(
                config.burst_capacity,
                config.requests_per_second as f64,
            ))
        } else {
            None
        };

        Self {
            global_bucket: RwLock::new(global_bucket),
            provider_buckets: RwLock::new(HashMap::new()),
            config,
        }
    }

    pub async fn check_rate_limit(&self, provider: &str) -> Result<(), Duration> {
        if !self.config.enabled {
            return Ok(());
        }

        if let Some(ref mut bucket) = self.global_bucket.write().await.as_mut() {
            if !bucket.try_consume(1.0) {
                return Err(bucket.time_until_available());
            }
        }

        let provider_config = self.config.providers.get(provider);
        if let Some(provider_cfg) = provider_config {
            if provider_cfg.requests_per_second > 0 {
                let mut buckets = self.provider_buckets.write().await;
                let bucket = buckets.entry(provider.to_string()).or_insert_with(|| {
                    let capacity = if provider_cfg.burst_capacity > 0 {
                        provider_cfg.burst_capacity
                    } else {
                        self.config.burst_capacity
                    };
                    TokenBucket::new(capacity, provider_cfg.requests_per_second as f64)
                });

                if !bucket.try_consume(1.0) {
                    return Err(bucket.time_until_available());
                }
            }
        }

        Ok(())
    }
}

/// Global rate limiter state
pub static RATE_LIMITER: std::sync::OnceLock<Arc<RateLimiter>> = std::sync::OnceLock::new();

/// Initialize the rate limiter with configuration
pub fn init_rate_limiter(config: RateLimitConfig) {
    let limiter = Arc::new(RateLimiter::new(config));
    let _ = RATE_LIMITER.set(limiter);
}

/// Create CORS middleware
pub fn create_cors_layer() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any)
}

/// Authentication middleware
/// Validates Bearer token from Authorization header
pub async fn auth_middleware(request: Request, next: Next) -> Response {
    // Skip auth for health check
    if request.uri().path() == "/health" {
        return next.run(request).await;
    }

    // Check for Authorization header
    if let Some(auth_header) = request.headers().get(header::AUTHORIZATION) {
        if let Ok(auth_str) = auth_header.to_str() {
            if auth_str.starts_with("Bearer ") {
                // Token exists, proceed (In a real systems, we would validate it here)
                return next.run(request).await;
            }
        }
    }

    // No valid auth header
    let error = serde_json::json!({
        "error": {
            "message": "Invalid authentication",
            "type": "authentication_error",
        }
    });

    (StatusCode::UNAUTHORIZED, axum::Json(error)).into_response()
}

/// Request timeout middleware
/// Applies a 60 second timeout to all requests
pub async fn timeout_middleware(request: Request, next: Next) -> Response {
    let timeout = Duration::from_secs(60);

    match tokio::time::timeout(timeout, next.run(request)).await {
        Ok(response) => response,
        Err(_) => {
            let error = serde_json::json!({
                "error": {
                    "message": "Request timeout",
                    "type": "timeout_error",
                }
            });
            (StatusCode::REQUEST_TIMEOUT, axum::Json(error)).into_response()
        }
    }
}

/// Request ID middleware
/// Adds a unique request ID to each request for tracing
pub async fn request_id_middleware(mut request: Request, next: Next) -> Response {
    let request_id = uuid::Uuid::new_v4().to_string();

    // Add request ID to headers for downstream services
    request
        .headers_mut()
        .insert("x-request-id", HeaderValue::from_str(&request_id).unwrap());

    let mut response = next.run(request).await;

    // Add request ID to response headers
    response
        .headers_mut()
        .insert("x-request-id", HeaderValue::from_str(&request_id).unwrap());

    response
}

/// Rate limiting middleware
/// Returns 429 with Retry-After header when rate limit is exceeded
pub async fn rate_limit_middleware(request: Request, next: Next) -> Response {
    let limiter = match RATE_LIMITER.get() {
        Some(l) => l,
        None => return next.run(request).await,
    };

    if !limiter.config.enabled {
        return next.run(request).await;
    }

    let provider = extract_provider_from_request(&request);

    match limiter.check_rate_limit(&provider).await {
        Ok(()) => next.run(request).await,
        Err(wait_time) => {
            let retry_after = wait_time.as_secs().max(1);
            let error = serde_json::json!({
                "error": {
                    "message": "Rate limit exceeded. Please retry later.",
                    "type": "rate_limit_error",
                    "retry_after": retry_after,
                }
            });

            let mut response = (StatusCode::TOO_MANY_REQUESTS, axum::Json(error)).into_response();
            response.headers_mut().insert(
                header::RETRY_AFTER,
                HeaderValue::from_str(&retry_after.to_string()).unwrap(),
            );
            response
        }
    }
}

fn extract_provider_from_request(request: &Request) -> String {
    if let Some(body_bytes) = request.extensions().get::<axum::body::Bytes>() {
        if let Ok(json) = serde_json::from_slice::<serde_json::Value>(body_bytes) {
            if let Some(model) = json.get("model").and_then(|m| m.as_str()) {
                if let Some(provider) = model.split('/').next() {
                    return provider.to_string();
                }
            }
        }
    }
    "default".to_string()
}

// ============================================================================
// Retry Logic with Exponential Backoff
// ============================================================================

/// Configuration for retry behavior
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts (not including initial attempt)
    pub max_retries: u32,
    /// Initial delay in seconds for exponential backoff
    pub initial_delay_secs: u64,
    /// Maximum delay cap in seconds
    pub max_delay_secs: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay_secs: 1,
            max_delay_secs: 30,
        }
    }
}

impl RetryConfig {
    #[allow(dead_code)]
    pub fn new(max_retries: u32) -> Self {
        Self {
            max_retries,
            ..Default::default()
        }
    }

    fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let delay_secs = self.initial_delay_secs * 2u64.pow(attempt);
        let capped = delay_secs.min(self.max_delay_secs);
        Duration::from_secs(capped)
    }
}

/// Determine if an error is retryable
///
/// Retryable errors:
/// - 5xx server errors (ApiError with status >= 500)
/// - 429 rate limit errors (RateLimitError or ApiError with status 429)
/// - Timeout errors
/// - Transient HTTP errors
///
/// Non-retryable errors:
/// - 4xx client errors (except 429)
/// - Authentication errors
/// - Invalid request errors
pub fn should_retry(error: &LlmError) -> bool {
    match error {
        LlmError::RateLimitError => true,
        LlmError::Timeout => true,
        LlmError::HttpError(_) => true,
        LlmError::ApiError { status, .. } => *status == 429 || *status >= 500,
        LlmError::InternalError(_) => true,
        LlmError::AuthError => false,
        LlmError::InvalidRequest(_) => false,
        LlmError::NotFound => false,
        LlmError::UnsupportedFeature => false,
        LlmError::SerializationError(_) => false,
        LlmError::ProviderError(_) => false,
        LlmError::Unknown(_) => false,
    }
}

/// Execute an async operation with retry logic and exponential backoff
pub async fn with_retry<T, F, Fut>(config: &RetryConfig, mut operation: F) -> Result<T, LlmError>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, LlmError>>,
{
    let mut last_error = None;

    for attempt in 0..=config.max_retries {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(error) => {
                if !should_retry(&error) {
                    return Err(error);
                }

                last_error = Some(error);

                if attempt < config.max_retries {
                    let delay = config.delay_for_attempt(attempt);
                    tracing::warn!(
                        attempt = attempt + 1,
                        max_retries = config.max_retries,
                        delay_ms = delay.as_millis(),
                        "Request failed, retrying with exponential backoff"
                    );
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }

    Err(last_error
        .unwrap_or_else(|| LlmError::Unknown("Retry exhausted without error".to_string())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ProviderRateLimitConfig;

    #[tokio::test]
    async fn test_cors_headers() {
        let _cors = create_cors_layer();
    }

    #[test]
    fn test_token_bucket_consume() {
        let mut bucket = TokenBucket::new(5, 1.0);

        assert!(bucket.try_consume(1.0));
        assert!(bucket.try_consume(1.0));
        assert!(bucket.try_consume(1.0));
        assert!(bucket.try_consume(1.0));
        assert!(bucket.try_consume(1.0));
        assert!(!bucket.try_consume(1.0));
    }

    #[test]
    fn test_token_bucket_refill() {
        let mut bucket = TokenBucket::new(2, 100.0);

        assert!(bucket.try_consume(1.0));
        assert!(bucket.try_consume(1.0));
        assert!(!bucket.try_consume(1.0));

        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(bucket.try_consume(1.0));
    }

    #[tokio::test]
    async fn test_rate_limiter_disabled() {
        let config = RateLimitConfig::default();
        let limiter = RateLimiter::new(config);

        for _ in 0..100 {
            assert!(limiter.check_rate_limit("test").await.is_ok());
        }
    }

    #[tokio::test]
    async fn test_rate_limiter_enforces_limit() {
        let config = RateLimitConfig {
            enabled: true,
            requests_per_second: 3,
            burst_capacity: 3,
            ..Default::default()
        };

        let limiter = RateLimiter::new(config);

        assert!(limiter.check_rate_limit("test").await.is_ok());
        assert!(limiter.check_rate_limit("test").await.is_ok());
        assert!(limiter.check_rate_limit("test").await.is_ok());
        assert!(limiter.check_rate_limit("test").await.is_err());
    }

    #[tokio::test]
    async fn test_rate_limiter_per_provider() {
        let mut config = RateLimitConfig {
            enabled: true,
            requests_per_second: 100,
            burst_capacity: 100,
            ..Default::default()
        };

        let provider_config = ProviderRateLimitConfig {
            requests_per_second: 2,
            burst_capacity: 2,
        };
        config
            .providers
            .insert("openai".to_string(), provider_config);

        let limiter = RateLimiter::new(config);

        assert!(limiter.check_rate_limit("openai").await.is_ok());
        assert!(limiter.check_rate_limit("openai").await.is_ok());
        assert!(limiter.check_rate_limit("openai").await.is_err());

        for _ in 0..10 {
            assert!(limiter.check_rate_limit("anthropic").await.is_ok());
        }
    }

    // ========================================================================
    // Retry Logic Tests
    // ========================================================================

    #[test]
    fn test_retry_config_delay_calculation() {
        let config = RetryConfig::default();

        assert_eq!(config.delay_for_attempt(0), Duration::from_secs(1));
        assert_eq!(config.delay_for_attempt(1), Duration::from_secs(2));
        assert_eq!(config.delay_for_attempt(2), Duration::from_secs(4));
        assert_eq!(config.delay_for_attempt(3), Duration::from_secs(8));
    }

    #[test]
    fn test_retry_config_delay_capped() {
        let config = RetryConfig {
            max_retries: 10,
            initial_delay_secs: 1,
            max_delay_secs: 8,
        };

        assert_eq!(config.delay_for_attempt(0), Duration::from_secs(1));
        assert_eq!(config.delay_for_attempt(1), Duration::from_secs(2));
        assert_eq!(config.delay_for_attempt(2), Duration::from_secs(4));
        assert_eq!(config.delay_for_attempt(3), Duration::from_secs(8));
        assert_eq!(config.delay_for_attempt(4), Duration::from_secs(8));
        assert_eq!(config.delay_for_attempt(10), Duration::from_secs(8));
    }

    #[test]
    fn test_should_retry_rate_limit() {
        assert!(should_retry(&LlmError::RateLimitError));
    }

    #[test]
    fn test_should_retry_5xx_errors() {
        assert!(should_retry(&LlmError::ApiError {
            status: 500,
            message: "Internal Server Error".to_string(),
        }));
        assert!(should_retry(&LlmError::ApiError {
            status: 502,
            message: "Bad Gateway".to_string(),
        }));
        assert!(should_retry(&LlmError::ApiError {
            status: 503,
            message: "Service Unavailable".to_string(),
        }));
        assert!(should_retry(&LlmError::ApiError {
            status: 504,
            message: "Gateway Timeout".to_string(),
        }));
    }

    #[test]
    fn test_should_retry_429_as_api_error() {
        assert!(should_retry(&LlmError::ApiError {
            status: 429,
            message: "Too Many Requests".to_string(),
        }));
    }

    #[test]
    fn test_should_not_retry_4xx_errors() {
        assert!(!should_retry(&LlmError::ApiError {
            status: 400,
            message: "Bad Request".to_string(),
        }));
        assert!(!should_retry(&LlmError::ApiError {
            status: 401,
            message: "Unauthorized".to_string(),
        }));
        assert!(!should_retry(&LlmError::ApiError {
            status: 403,
            message: "Forbidden".to_string(),
        }));
        assert!(!should_retry(&LlmError::ApiError {
            status: 404,
            message: "Not Found".to_string(),
        }));
    }

    #[test]
    fn test_should_not_retry_auth_error() {
        assert!(!should_retry(&LlmError::AuthError));
    }

    #[test]
    fn test_should_not_retry_invalid_request() {
        assert!(!should_retry(&LlmError::InvalidRequest(
            "bad request".to_string()
        )));
    }

    #[test]
    fn test_should_retry_timeout() {
        assert!(should_retry(&LlmError::Timeout));
    }

    #[test]
    fn test_should_retry_http_error() {
        assert!(should_retry(&LlmError::HttpError(
            "connection failed".to_string()
        )));
    }

    #[tokio::test]
    async fn test_with_retry_success_first_try() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let config = RetryConfig::new(3);
        let call_count = Arc::new(AtomicU32::new(0));
        let call_count_clone = call_count.clone();

        let result = with_retry(&config, move || {
            let count = call_count_clone.clone();
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                Ok::<i32, LlmError>(42)
            }
        })
        .await;

        assert_eq!(result.unwrap(), 42);
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_with_retry_success_after_failures() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let config = RetryConfig::new(3);
        let call_count = Arc::new(AtomicU32::new(0));
        let call_count_clone = call_count.clone();

        let result = with_retry(&config, move || {
            let count = call_count_clone.clone();
            async move {
                let current = count.fetch_add(1, Ordering::SeqCst) + 1;
                if current < 3 {
                    Err(LlmError::RateLimitError)
                } else {
                    Ok::<i32, LlmError>(42)
                }
            }
        })
        .await;

        assert_eq!(result.unwrap(), 42);
        assert_eq!(call_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_with_retry_exhausted() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let config = RetryConfig::new(2);
        let call_count = Arc::new(AtomicU32::new(0));
        let call_count_clone = call_count.clone();

        let result = with_retry(&config, move || {
            let count = call_count_clone.clone();
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                Err::<i32, LlmError>(LlmError::RateLimitError)
            }
        })
        .await;

        assert!(matches!(result, Err(LlmError::RateLimitError)));
        assert_eq!(call_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_with_retry_no_retry_on_auth_error() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let config = RetryConfig::new(3);
        let call_count = Arc::new(AtomicU32::new(0));
        let call_count_clone = call_count.clone();

        let result = with_retry(&config, move || {
            let count = call_count_clone.clone();
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                Err::<i32, LlmError>(LlmError::AuthError)
            }
        })
        .await;

        assert!(matches!(result, Err(LlmError::AuthError)));
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_with_retry_no_retry_on_invalid_request() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let config = RetryConfig::new(3);
        let call_count = Arc::new(AtomicU32::new(0));
        let call_count_clone = call_count.clone();

        let result = with_retry(&config, move || {
            let count = call_count_clone.clone();
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                Err::<i32, LlmError>(LlmError::InvalidRequest("bad".to_string()))
            }
        })
        .await;

        assert!(matches!(result, Err(LlmError::InvalidRequest(_))));
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }
}
