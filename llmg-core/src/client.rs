use crate::provider::Credentials;
use std::time::Duration;

/// HTTP client for making requests to LLM providers
#[derive(Debug)]
pub struct LlmClient {
    inner: reqwest::Client,
    base_url: String,
    credentials: Option<Box<dyn Credentials>>,
}

impl LlmClient {
    /// Create a new LLM client with default configuration
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            inner: reqwest::Client::new(),
            base_url: base_url.into(),
            credentials: None,
        }
    }

    /// Get the underlying reqwest client
    pub fn inner(&self) -> &reqwest::Client {
        &self.inner
    }

    /// Get the base URL
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Set credentials for authentication
    pub fn with_credentials(mut self, credentials: Box<dyn Credentials>) -> Self {
        self.credentials = Some(credentials);
        self
    }

    /// Build a request with authentication applied
    pub fn request(
        &self,
        method: reqwest::Method,
        path: &str,
    ) -> Result<reqwest::Request, crate::provider::LlmError> {
        let url = format!("{}{}", self.base_url, path);
        let mut req = reqwest::Request::new(
            method,
            url.parse().map_err(|e| {
                crate::provider::LlmError::InvalidRequest(format!("Invalid URL: {}", e))
            })?,
        );

        if let Some(creds) = &self.credentials {
            creds.apply(&mut req)?;
        }

        Ok(req)
    }
}

/// Builder for LlmClient
#[derive(Debug)]
pub struct ClientBuilder {
    timeout: Duration,
    pool_idle_timeout: Duration,
    pool_max_idle_per_host: usize,
    base_url: String,
    credentials: Option<Box<dyn Credentials>>,
}

impl ClientBuilder {
    /// Create a new client builder
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            timeout: Duration::from_secs(60),
            pool_idle_timeout: Duration::from_secs(90),
            pool_max_idle_per_host: 32,
            base_url: base_url.into(),
            credentials: None,
        }
    }

    /// Set request timeout
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set pool idle timeout
    pub fn pool_idle_timeout(mut self, timeout: Duration) -> Self {
        self.pool_idle_timeout = timeout;
        self
    }

    /// Set max idle connections per host
    pub fn pool_max_idle_per_host(mut self, max: usize) -> Self {
        self.pool_max_idle_per_host = max;
        self
    }

    /// Set credentials
    pub fn credentials(mut self, credentials: Box<dyn Credentials>) -> Self {
        self.credentials = Some(credentials);
        self
    }

    /// Build the client
    pub fn build(self) -> reqwest::Result<LlmClient> {
        let client = reqwest::Client::builder()
            .timeout(self.timeout)
            .pool_idle_timeout(self.pool_idle_timeout)
            .pool_max_idle_per_host(self.pool_max_idle_per_host)
            .build()?;

        Ok(LlmClient {
            inner: client,
            base_url: self.base_url,
            credentials: self.credentials,
        })
    }
}

impl Default for ClientBuilder {
    fn default() -> Self {
        Self::new("https://api.openai.com/v1")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_builder_defaults() {
        let builder = ClientBuilder::default();
        assert_eq!(builder.timeout, Duration::from_secs(60));
        assert_eq!(builder.pool_max_idle_per_host, 32);
    }

    #[test]
    fn test_client_builder_custom() {
        let builder = ClientBuilder::new("https://api.example.com")
            .timeout(Duration::from_secs(30))
            .pool_max_idle_per_host(10);

        assert_eq!(builder.timeout, Duration::from_secs(30));
        assert_eq!(builder.pool_max_idle_per_host, 10);
        assert_eq!(builder.base_url, "https://api.example.com");
    }
}
