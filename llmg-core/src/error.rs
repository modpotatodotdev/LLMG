//! Error types for LLMG
//!
//! These types are compatible with the OpenAI API error format.

use serde::{Deserialize, Serialize};

/// OpenAI-compatible error response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// The error object
    pub error: ErrorDetail,
}

/// Detailed error information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetail {
    /// Error message
    pub message: String,
    /// Error type
    pub r#type: String,
    /// Parameter that caused the error (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    /// Error code
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

impl ErrorResponse {
    /// Create a new error response
    pub fn new(message: impl Into<String>, error_type: impl Into<String>) -> Self {
        Self {
            error: ErrorDetail {
                message: message.into(),
                r#type: error_type.into(),
                param: None,
                code: None,
            },
        }
    }

    /// Create an invalid request error
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::new(message, "invalid_request_error")
    }

    /// Create an authentication error
    pub fn authentication_error() -> Self {
        Self::new("Invalid authentication", "authentication_error")
    }

    /// Create a rate limit error
    pub fn rate_limit_error() -> Self {
        Self::new("Rate limit exceeded", "rate_limit_error")
    }

    /// Create a server error
    pub fn server_error(message: impl Into<String>) -> Self {
        Self::new(message, "server_error")
    }
}

/// Normalize provider errors to OpenAI format
pub fn normalize_error(status: u16, message: impl Into<String>) -> ErrorResponse {
    let message = message.into();
    match status {
        400 => ErrorResponse::invalid_request(message),
        401 => ErrorResponse::authentication_error(),
        429 => ErrorResponse::rate_limit_error(),
        _ => ErrorResponse::server_error(message),
    }
}

impl std::fmt::Display for ErrorResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.error.r#type, self.error.message)
    }
}

impl std::error::Error for ErrorResponse {}
