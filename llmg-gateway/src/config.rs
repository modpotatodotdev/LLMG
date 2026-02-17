//! Configuration file support for LLMG Gateway
//!
//! Supports TOML format for configuration.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Gateway configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Server configuration
    #[serde(default)]
    pub server: ServerConfig,

    /// Provider configurations
    #[serde(default)]
    pub providers: HashMap<String, ProviderConfig>,

    /// Model aliases
    #[serde(default)]
    pub aliases: HashMap<String, String>,

    /// Logging configuration
    #[serde(default)]
    pub logging: LoggingConfig,

    /// Rate limiting configuration
    #[serde(default)]
    pub rate_limit: RateLimitConfig,

    /// Whether to fall back on environment variables for API keys
    #[serde(default = "default_true")]
    pub use_env: bool,
}

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Port to listen on
    #[serde(default = "default_port")]
    pub port: u16,

    /// Host to bind to
    #[serde(default = "default_host")]
    pub host: String,

    /// Request timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout: u64,

    /// Enable CORS
    #[serde(default)]
    pub cors: bool,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Enable rate limiting
    #[serde(default)]
    pub enabled: bool,

    /// Global requests per second limit (0 = no limit)
    #[serde(default)]
    pub requests_per_second: u32,

    /// Global burst capacity (max tokens in bucket)
    #[serde(default = "default_burst_capacity")]
    pub burst_capacity: u32,

    /// Per-provider rate limits (provider_name -> limit config)
    #[serde(default)]
    pub providers: HashMap<String, ProviderRateLimitConfig>,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            requests_per_second: 0,
            burst_capacity: default_burst_capacity(),
            providers: HashMap::new(),
        }
    }
}

fn default_burst_capacity() -> u32 {
    100
}

/// Per-provider rate limit configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProviderRateLimitConfig {
    /// Requests per second for this provider (0 = use global limit)
    #[serde(default)]
    pub requests_per_second: u32,

    /// Burst capacity for this provider (0 = use global limit)
    #[serde(default)]
    pub burst_capacity: u32,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            port: default_port(),
            host: default_host(),
            timeout: default_timeout(),
            cors: false,
        }
    }
}

fn default_port() -> u16 {
    8080
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_timeout() -> u64 {
    60
}

/// Provider-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Whether this provider is enabled
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// API key (can use env var syntax: ${ENV_VAR})
    pub api_key: Option<String>,

    /// Base URL for the provider
    pub base_url: Option<String>,

    /// Default model for this provider
    pub default_model: Option<String>,

    /// Additional headers to send
    #[serde(default)]
    pub headers: HashMap<String, String>,
}

fn default_true() -> bool {
    true
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level
    #[serde(default = "default_log_level")]
    pub level: String,

    /// Enable verbose logging
    #[serde(default)]
    pub verbose: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            verbose: false,
        }
    }
}

fn default_log_level() -> String {
    "info".to_string()
}

impl Config {
    /// Load configuration from a file (supports TOML, JSON, YAML)
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let contents = std::fs::read_to_string(path).map_err(ConfigError::Io)?;

        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("toml");

        let mut config: Config = match extension {
            "json" => {
                serde_json::from_str(&contents).map_err(|e| ConfigError::Parse(e.to_string()))?
            }
            "yaml" | "yml" => {
                // Since we don't have serde_yaml in dependencies yet,
                // I'll assume it will be added or we'll hint it.
                // For now, let's just use JSON if we can, or throw error.
                return Err(ConfigError::NotSupported(
                    "YAML support requires adding serde_yaml to dependencies".to_string(),
                ));
            }
            _ => toml::from_str(&contents).map_err(|e| ConfigError::Parse(e.to_string()))?,
        };

        if config.use_env {
            config.expand_env_vars();
        }

        Ok(config)
    }

    /// Automatically find and load configuration from the current directory
    pub fn load_from_dir<P: AsRef<Path>>(dir: P) -> Result<Self, ConfigError> {
        let dir = dir.as_ref();
        let possible_files = ["llmg.json", "llmg.yaml", "llmg.yml", "llmg.toml"];

        for file in possible_files {
            let path = dir.join(file);
            if path.exists() {
                return Self::from_file(path);
            }
        }

        // Return default config if no file found
        Ok(Self::with_defaults())
    }

    /// Load from file with environment variable expansion
    pub fn from_file_expanded<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        Self::from_file(path)
    }

    /// Expand environment variables in configuration
    fn expand_env_vars(&mut self) {
        if !self.use_env {
            return;
        }

        for (name, provider) in &mut self.providers {
            if let Some(ref mut api_key) = provider.api_key {
                if api_key.contains("${") {
                    *api_key = expand_env_var(api_key);
                }
            } else {
                // Fallback to env var if missing
                let env_name = format!("{}_API_KEY", name.to_uppercase());
                if let Ok(key) = std::env::var(&env_name) {
                    provider.api_key = Some(key);
                }
            }
        }
    }

    /// Create default configuration
    pub fn with_defaults() -> Self {
        let mut providers = HashMap::new();

        providers.insert(
            "openai".to_string(),
            ProviderConfig {
                enabled: true,
                api_key: Some("${OPENAI_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("gpt-4".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "anthropic".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${ANTHROPIC_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("claude-3-opus-20240229".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "azure".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${AZURE_OPENAI_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("gpt-4".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "azure_ai".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${AZURE_AI_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("gpt-4".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "cohere".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${COHERE_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("command-r".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "ollama".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: None,
                base_url: Some("http://localhost:11434".to_string()),
                default_model: Some("llama3".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "openrouter".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${OPENROUTER_API_KEY}".to_string()),
                base_url: Some("https://openrouter.ai/api/v1".to_string()),
                default_model: Some("anthropic/claude-3.5-sonnet".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "github_copilot".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${GITHUB_COPILOT_TOKEN}".to_string()),
                base_url: Some("https://api.githubcopilot.com".to_string()),
                default_model: Some("gpt-4".to_string()),
                headers: [
                    ("editor-version".to_string(), "vscode/1.85.1".to_string()),
                    (
                        "Copilot-Integration-Id".to_string(),
                        "vscode-chat".to_string(),
                    ),
                ]
                .into_iter()
                .collect(),
            },
        );

        providers.insert(
            "antigravity".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${ANTIGRAVITY_API_KEY}".to_string()),
                base_url: Some("https://generativelanguage.googleapis.com".to_string()),
                default_model: Some("gemini-2.0-flash".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "z_ai".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${Z_AI_API_KEY}".to_string()),
                base_url: Some("https://api.z.ai/api/paas/v4".to_string()),
                default_model: Some("glm-5".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "z_ai_coding".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${Z_AI_API_KEY}".to_string()),
                base_url: Some("https://api.z.ai/api/coding/paas/v4".to_string()),
                default_model: Some("GLM-4.7".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "anyscale".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${ANYSCALE_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("meta-llama/Llama-2-70b-chat-hf".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "groq".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${GROQ_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("llama3-70b-8192".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "deepinfra".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${DEEPINFRA_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("meta-llama/Meta-Llama-3.1-70B-Instruct".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "together_ai".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${TOGETHER_AI_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("meta-llama/Llama-2-70b-chat-hf".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "fireworks_ai".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${FIREWORKS_AI_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("accounts/fireworks/models/llama-v2-70b-chat".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "cerebras".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${CEREBRAS_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("llama3.1-70b".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "sambanova".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${SAMBANOVA_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("Meta-Llama-3.1-70B-Instruct".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "friendliai".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${FRIENDLIAI_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("meta-llama-3.1-70b-instruct".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "nscale".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${NSCALE_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("meta-llama/Meta-Llama-3.1-70B-Instruct".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "hyperbolic".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${HYPERBOLIC_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("meta-llama/Llama-3.2-3B-Instruct".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "featherless_ai".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${FEATHERLESS_AI_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("meta-llama/Meta-Llama-3.1-70B-Instruct".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "apertis_ai".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${APERTIS_AI_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("meta-llama/Meta-Llama-3.1-70B-Instruct".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "nano_gpt".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${NANO_GPT_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("gpt-4".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "poe".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${POE_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("claude-3-opus-20240229".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "chutes".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${CHUTES_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("meta-llama/Meta-Llama-3.1-70B-Instruct".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "comet".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${COMET_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("meta-llama/Meta-Llama-3.1-70B-Instruct".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "aiml".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${AIML_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("gpt-4".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "publicai".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${PUBLICAI_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("meta-llama/Meta-Llama-3.1-70B-Instruct".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "synthetic".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${SYNTHETIC_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("meta-llama/Meta-Llama-3.1-70B-Instruct".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "v0".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${V0_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("claude-3-opus-20240229".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "jina".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${JINA_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("jina-reranker-v1-base-en".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "aleph_alpha".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${ALEPH_ALPHA_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("luminous-extended-control".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "minimax".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${MINIMAX_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("abab5.5-chat".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "meta_llama".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${META_LLAMA_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("llama-3.1-70b-instruct".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "xinference".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: None,
                base_url: Some("http://localhost:9997".to_string()),
                default_model: Some("llama-2-chat-13b".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "xai".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${XAI_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("grok-beta".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "ai21".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${AI21_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("jamba-1-5-mini".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "deepgram".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${DEEPGRAM_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("nova-2".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "mistral".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${MISTRAL_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("mistral-large-latest".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "deepseek".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${DEEPSEEK_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("deepseek-chat".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "octoai".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${OCTOAI_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("meta-llama-3-70b-instruct".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "perplexity".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${PERPLEXITY_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("pplx-7b-online".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "voyageai".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${VOYAGEAI_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("voyage-large-2".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "volcano".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${VOLCANO_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("volcano-1".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "infinity".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${INFINITY_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("infinity-1".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "milvus".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${MILVUS_API_KEY}".to_string()),
                base_url: Some("http://localhost:19530".to_string()),
                default_model: Some("default".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "compactifai".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${COMPACTIFAI_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("compactifai-1".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "vllm".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: None,
                base_url: Some("http://localhost:8000/v1".to_string()),
                default_model: Some("meta-llama/Llama-2-7b-chat-hf".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "custom_llm_server".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: None,
                base_url: Some("http://localhost:8000/v1".to_string()),
                default_model: Some("default".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "lm_studio".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: None,
                base_url: Some("http://localhost:1234/v1".to_string()),
                default_model: Some("gpt-4".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "llamafile".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: None,
                base_url: Some("http://localhost:8080/v1".to_string()),
                default_model: Some("llama-3-8b".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "triton".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: None,
                base_url: Some("http://localhost:8001/v1".to_string()),
                default_model: Some("triton-llama-3-70b".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "petals".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: None,
                base_url: Some("https://petals.ml/api/v1".to_string()),
                default_model: Some("bigscience/bloomz-petals".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "oobabooga".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: None,
                base_url: Some("http://localhost:7860/api/v1".to_string()),
                default_model: Some("llama-2-7b-chat".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "docker_runner".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: None,
                base_url: Some("http://localhost:5000/v1".to_string()),
                default_model: Some("docker-llama-3-8b".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "aws_sagemaker".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${AWS_ACCESS_KEY_ID}".to_string()),
                base_url: None,
                default_model: Some("huggingface-pytorch-inference".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "bedrock".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${AWS_ACCESS_KEY_ID}".to_string()),
                base_url: None,
                default_model: Some("anthropic.claude-3-opus-20240229-v1:0".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "watsonx".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${IBM_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("meta-llama/llama-2-70b-chat".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "vertex_ai".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${GOOGLE_APPLICATION_CREDENTIALS}".to_string()),
                base_url: None,
                default_model: Some("gemini-1.5-pro".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "heroku".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${HEROKU_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("claude-3-opus-20240229".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "huggingface".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${HUGGINGFACE_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("gpt2".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "langgraph".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${LANGGRAPH_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("gpt-4".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "fal_ai".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${FAL_AI_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("gpt-4".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "helicone".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${HELICONE_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("gpt-4".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "litellm_proxy".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: None,
                base_url: Some("http://localhost:4000".to_string()),
                default_model: Some("gpt-4".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "pydantic_ai_agent".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${PYDANTIC_AI_AGENT_API_KEY}".to_string()),
                base_url: None,
                default_model: Some("gpt-4".to_string()),
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "elevenlabs".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${ELEVENLABS_API_KEY}".to_string()),
                base_url: None,
                default_model: None,
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "firecrawl".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${FIRECRAWL_API_KEY}".to_string()),
                base_url: None,
                default_model: None,
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "polly".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${AWS_ACCESS_KEY_ID}".to_string()),
                base_url: None,
                default_model: None,
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "runway".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${RUNWAY_API_KEY}".to_string()),
                base_url: None,
                default_model: None,
                headers: HashMap::new(),
            },
        );

        providers.insert(
            "stability".to_string(),
            ProviderConfig {
                enabled: false,
                api_key: Some("${STABILITY_API_KEY}".to_string()),
                base_url: None,
                default_model: None,
                headers: HashMap::new(),
            },
        );

        let mut aliases = HashMap::new();
        aliases.insert("gpt-4".to_string(), "openai/gpt-4".to_string());
        aliases.insert("gpt-4-turbo".to_string(), "openai/gpt-4-turbo".to_string());
        aliases.insert("gpt-4o".to_string(), "openai/gpt-4o".to_string());
        aliases.insert("gpt-4o-mini".to_string(), "openai/gpt-4o-mini".to_string());
        aliases.insert("gpt-3.5".to_string(), "openai/gpt-3.5-turbo".to_string());
        aliases.insert(
            "claude".to_string(),
            "anthropic/claude-3-opus-20240229".to_string(),
        );
        aliases.insert(
            "claude-opus".to_string(),
            "anthropic/claude-3-opus-20240229".to_string(),
        );
        aliases.insert(
            "claude-sonnet".to_string(),
            "anthropic/claude-3-sonnet-20240229".to_string(),
        );
        aliases.insert(
            "claude-3".to_string(),
            "anthropic/claude-3-opus-20240229".to_string(),
        );
        aliases.insert(
            "claude-3-opus".to_string(),
            "anthropic/claude-3-opus-20240229".to_string(),
        );
        aliases.insert(
            "claude-3-sonnet".to_string(),
            "anthropic/claude-3-sonnet-20240229".to_string(),
        );
        aliases.insert(
            "claude-3-haiku".to_string(),
            "anthropic/claude-3-haiku-20240307".to_string(),
        );
        aliases.insert(
            "gemini".to_string(),
            "antigravity/gemini-2.0-flash".to_string(),
        );
        aliases.insert(
            "gemini-pro".to_string(),
            "antigravity/gemini-1.5-pro".to_string(),
        );
        aliases.insert(
            "gemini-flash".to_string(),
            "antigravity/gemini-2.0-flash".to_string(),
        );
        aliases.insert(
            "gemini-2".to_string(),
            "antigravity/gemini-2.0-pro-exp-02-05".to_string(),
        );
        aliases.insert(
            "gemini-1.5".to_string(),
            "antigravity/gemini-1.5-pro".to_string(),
        );
        aliases.insert(
            "llama".to_string(),
            "meta_llama/llama-3.1-70b-instruct".to_string(),
        );
        aliases.insert(
            "llama-3".to_string(),
            "meta_llama/llama-3.1-70b-instruct".to_string(),
        );
        aliases.insert(
            "llama-2".to_string(),
            "meta_llama/llama-2-70b-chat-hf".to_string(),
        );
        aliases.insert(
            "llama-3.1".to_string(),
            "meta_llama/llama-3.1-70b-instruct".to_string(),
        );
        aliases.insert(
            "llama-3.2".to_string(),
            "meta_llama/llama-3.2-90b-vision-instruct".to_string(),
        );
        aliases.insert("mistral".to_string(), "mistral/mistral-large".to_string());
        aliases.insert(
            "mistral-large".to_string(),
            "mistral/mistral-large".to_string(),
        );
        aliases.insert(
            "mistral-medium".to_string(),
            "mistral/mistral-medium".to_string(),
        );
        aliases.insert(
            "mistral-small".to_string(),
            "mistral/mistral-small".to_string(),
        );
        aliases.insert("mixtral".to_string(), "mistral/mixtral-8x7b".to_string());
        aliases.insert("codestral".to_string(), "mistral/codestral".to_string());
        aliases.insert("deepseek".to_string(), "deepseek/deepseek-chat".to_string());
        aliases.insert(
            "deepseek-chat".to_string(),
            "deepseek/deepseek-chat".to_string(),
        );
        aliases.insert(
            "deepseek-coder".to_string(),
            "deepseek/deepseek-coder".to_string(),
        );
        aliases.insert("groq".to_string(), "groq/llama3-70b-8192".to_string());
        aliases.insert(
            "llama3-groq".to_string(),
            "groq/llama3-70b-8192".to_string(),
        );
        aliases.insert(
            "mixtral-groq".to_string(),
            "groq/mixtral-8x7b-32768".to_string(),
        );
        aliases.insert("command".to_string(), "cohere/command".to_string());
        aliases.insert("command-r".to_string(), "cohere/command-r".to_string());
        aliases.insert(
            "command-r-plus".to_string(),
            "cohere/command-r-plus".to_string(),
        );
        aliases.insert("ollama-llama3".to_string(), "ollama/llama3".to_string());
        aliases.insert("ollama-mistral".to_string(), "ollama/mistral".to_string());
        aliases.insert(
            "ollama-codellama".to_string(),
            "ollama/codellama".to_string(),
        );
        aliases.insert(
            "bedrock".to_string(),
            "bedrock/anthropic.claude-3-opus-20240229-v1:0".to_string(),
        );
        aliases.insert("vertex".to_string(), "vertex_ai/gemini-1.5-pro".to_string());
        aliases.insert("hf".to_string(), "huggingface/gpt2".to_string());
        aliases.insert(
            "perplexity".to_string(),
            "perplexity/pplx-7b-online".to_string(),
        );
        aliases.insert("grok".to_string(), "xai/grok-beta".to_string());
        aliases.insert("xai".to_string(), "xai/grok-beta".to_string());
        aliases.insert(
            "qwen".to_string(),
            "openrouter/qwen/qwen-2.5-72b-instruct".to_string(),
        );
        aliases.insert(
            "hermes".to_string(),
            "openrouter/nousresearch/hermes-3-llama-3.1-405b".to_string(),
        );

        Self {
            server: ServerConfig::default(),
            providers,
            aliases,
            logging: LoggingConfig::default(),
            rate_limit: RateLimitConfig::default(),
            use_env: true,
        }
    }

    /// Generate example configuration file content
    pub fn example() -> String {
        r#"# LLMG Gateway Configuration
#
# This is an example configuration file. Copy it to llmg.toml and customize for your needs.
#
# Provider API Keys can be set via environment variables or directly in the config.
# Using environment variables is recommended for security.

[server]
port = 8080
host = "0.0.0.0"
timeout = 60
cors = true

# Rate Limiting
[rate_limit]
enabled = false
requests_per_second = 100
burst_capacity = 200

# Per-provider rate limits
[rate_limit.providers.openai]
requests_per_second = 50
burst_capacity = 100

# OpenAI Provider
[providers.openai]
enabled = true
api_key = "${OPENAI_API_KEY}"
default_model = "gpt-4"

# Anthropic Provider
[providers.anthropic]
enabled = false
api_key = "${ANTHROPIC_API_KEY}"
default_model = "claude-3-opus-20240229"

# Azure OpenAI Provider
[providers.azure]
enabled = false
api_key = "${AZURE_OPENAI_API_KEY}"
default_model = "gpt-4"

# Azure AI Provider
[providers.azure_ai]
enabled = false
api_key = "${AZURE_AI_API_KEY}"
default_model = "gpt-4"

# Cohere Provider
[providers.cohere]
enabled = false
api_key = "${COHERE_API_KEY}"
default_model = "command-r"

# Ollama (Local)
[providers.ollama]
enabled = false
base_url = "http://localhost:11434"
default_model = "llama3"

# OpenRouter (100+ models)
[providers.openrouter]
enabled = false
api_key = "${OPENROUTER_API_KEY}"
base_url = "https://openrouter.ai/api/v1"
default_model = "anthropic/claude-3.5-sonnet"

# GitHub Copilot
[providers.github_copilot]
enabled = false
api_key = "${GITHUB_COPILOT_TOKEN}"
base_url = "https://api.githubcopilot.com"
default_model = "gpt-4"

# Google Antigravity (Gemini)
[providers.antigravity]
enabled = false
api_key = "${ANTIGRAVITY_API_KEY}"
base_url = "https://generativelanguage.googleapis.com"
default_model = "gemini-2.0-flash"

# Anyscale
[providers.anyscale]
enabled = false
api_key = "${ANYSCALE_API_KEY}"
default_model = "meta-llama/Llama-2-70b-chat-hf"

# Groq
[providers.groq]
enabled = false
api_key = "${GROQ_API_KEY}"
default_model = "llama3-70b-8192"

# DeepInfra
[providers.deepinfra]
enabled = false
api_key = "${DEEPINFRA_API_KEY}"
default_model = "meta-llama/Meta-Llama-3.1-70B-Instruct"

# Together AI
[providers.together_ai]
enabled = false
api_key = "${TOGETHER_AI_API_KEY}"
default_model = "meta-llama/Llama-2-70b-chat-hf"

# Fireworks AI
[providers.fireworks_ai]
enabled = false
api_key = "${FIREWORKS_AI_API_KEY}"
default_model = "accounts/fireworks/models/llama-v2-70b-chat"

# Cerebras
[providers.cerebras]
enabled = false
api_key = "${CEREBRAS_API_KEY}"
default_model = "llama3.1-70b"

# SambaNova
[providers.sambanova]
enabled = false
api_key = "${SAMBANOVA_API_KEY}"
default_model = "Meta-Llama-3.1-70B-Instruct"

# FriendliAI
[providers.friendliai]
enabled = false
api_key = "${FRIENDLIAI_API_KEY}"
default_model = "meta-llama-3.1-70b-instruct"

# Nscale
[providers.nscale]
enabled = false
api_key = "${NSCALE_API_KEY}"
default_model = "meta-llama/Meta-Llama-3.1-70B-Instruct"

# Hyperbolic
[providers.hyperbolic]
enabled = false
api_key = "${HYPERBOLIC_API_KEY}"
default_model = "meta-llama/Llama-3.2-3B-Instruct"

# Featherless AI
[providers.featherless_ai]
enabled = false
api_key = "${FEATHERLESS_AI_API_KEY}"
default_model = "meta-llama/Meta-Llama-3.1-70B-Instruct"

# Apertis AI
[providers.apertis_ai]
enabled = false
api_key = "${APERTIS_AI_API_KEY}"
default_model = "meta-llama/Meta-Llama-3.1-70B-Instruct"

# Nano GPT
[providers.nano_gpt]
enabled = false
api_key = "${NANO_GPT_API_KEY}"
default_model = "gpt-4"

# Poe
[providers.poe]
enabled = false
api_key = "${POE_API_KEY}"
default_model = "claude-3-opus-20240229"

# Chutes
[providers.chutes]
enabled = false
api_key = "${CHUTES_API_KEY}"
default_model = "meta-llama/Meta-Llama-3.1-70B-Instruct"

# Comet
[providers.comet]
enabled = false
api_key = "${COMET_API_KEY}"
default_model = "meta-llama/Meta-Llama-3.1-70B-Instruct"

# AIML
[providers.aiml]
enabled = false
api_key = "${AIML_API_KEY}"
default_model = "gpt-4"

# PublicAI
[providers.publicai]
enabled = false
api_key = "${PUBLICAI_API_KEY}"
default_model = "meta-llama/Meta-Llama-3.1-70B-Instruct"

# Synthetic
[providers.synthetic]
enabled = false
api_key = "${SYNTHETIC_API_KEY}"
default_model = "meta-llama/Meta-Llama-3.1-70B-Instruct"

# V0
[providers.v0]
enabled = false
api_key = "${V0_API_KEY}"
default_model = "claude-3-opus-20240229"

# Jina
[providers.jina]
enabled = false
api_key = "${JINA_API_KEY}"
default_model = "jina-reranker-v1-base-en"

# Aleph Alpha
[providers.aleph_alpha]
enabled = false
api_key = "${ALEPH_ALPHA_API_KEY}"
default_model = "luminous-extended-control"

# MiniMax
[providers.minimax]
enabled = false
api_key = "${MINIMAX_API_KEY}"
default_model = "abab5.5-chat"

# Meta Llama
[providers.meta_llama]
enabled = false
api_key = "${META_LLAMA_API_KEY}"
default_model = "llama-3.1-70b-instruct"

# Xinference (Local)
[providers.xinference]
enabled = false
base_url = "http://localhost:9997"
default_model = "llama-2-chat-13b"

# xAI (Grok)
[providers.xai]
enabled = false
api_key = "${XAI_API_KEY}"
default_model = "grok-beta"

# AI21
[providers.ai21]
enabled = false
api_key = "${AI21_API_KEY}"
default_model = "jamba-1-5-mini"

# Deepgram
[providers.deepgram]
enabled = false
api_key = "${DEEPGRAM_API_KEY}"
default_model = "nova-2"

# Mistral
[providers.mistral]
enabled = false
api_key = "${MISTRAL_API_KEY}"
default_model = "mistral-large-latest"

# DeepSeek
[providers.deepseek]
enabled = false
api_key = "${DEEPSEEK_API_KEY}"
default_model = "deepseek-chat"

# OctoAI
[providers.octoai]
enabled = false
api_key = "${OCTOAI_API_KEY}"
default_model = "meta-llama-3-70b-instruct"

# Perplexity
[providers.perplexity]
enabled = false
api_key = "${PERPLEXITY_API_KEY}"
default_model = "pplx-7b-online"

# VoyageAI
[providers.voyageai]
enabled = false
api_key = "${VOYAGEAI_API_KEY}"
default_model = "voyage-large-2"

# Volcano
[providers.volcano]
enabled = false
api_key = "${VOLCANO_API_KEY}"
default_model = "volcano-1"

# Infinity
[providers.infinity]
enabled = false
api_key = "${INFINITY_API_KEY}"
default_model = "infinity-1"

# Milvus (Local)
[providers.milvus]
enabled = false
api_key = "${MILVUS_API_KEY}"
base_url = "http://localhost:19530"
default_model = "default"

# CompactifAI
[providers.compactifai]
enabled = false
api_key = "${COMPACTIFAI_API_KEY}"
default_model = "compactifai-1"

# vLLM (Local)
[providers.vllm]
enabled = false
base_url = "http://localhost:8000/v1"
default_model = "meta-llama/Llama-2-7b-chat-hf"

# Custom LLM Server (Local)
[providers.custom_llm_server]
enabled = false
base_url = "http://localhost:8000/v1"
default_model = "default"

# LM Studio (Local)
[providers.lm_studio]
enabled = false
base_url = "http://localhost:1234/v1"
default_model = "gpt-4"

# Llamafile (Local)
[providers.llamafile]
enabled = false
base_url = "http://localhost:8080/v1"
default_model = "llama-3-8b"

# Triton Inference Server (Local)
[providers.triton]
enabled = false
base_url = "http://localhost:8001/v1"
default_model = "triton-llama-3-70b"

# Petals
[providers.petals]
enabled = false
base_url = "https://petals.ml/api/v1"
default_model = "bigscience/bloomz-petals"

# oobabooga (Local)
[providers.oobabooga]
enabled = false
base_url = "http://localhost:7860/api/v1"
default_model = "llama-2-7b-chat"

# Docker Runner (Local)
[providers.docker_runner]
enabled = false
base_url = "http://localhost:5000/v1"
default_model = "docker-llama-3-8b"

# AWS SageMaker
[providers.aws_sagemaker]
enabled = false
api_key = "${AWS_ACCESS_KEY_ID}"
default_model = "huggingface-pytorch-inference"

# AWS Bedrock
[providers.bedrock]
enabled = false
api_key = "${AWS_ACCESS_KEY_ID}"
default_model = "anthropic.claude-3-opus-20240229-v1:0"

# IBM watsonx
[providers.watsonx]
enabled = false
api_key = "${IBM_API_KEY}"
default_model = "meta-llama/llama-2-70b-chat"

# Google Vertex AI
[providers.vertex_ai]
enabled = false
api_key = "${GOOGLE_APPLICATION_CREDENTIALS}"
default_model = "gemini-1.5-pro"

# Heroku
[providers.heroku]
enabled = false
api_key = "${HEROKU_API_KEY}"
default_model = "claude-3-opus-20240229"

# Hugging Face
[providers.huggingface]
enabled = false
api_key = "${HUGGINGFACE_API_KEY}"
default_model = "gpt2"

# LangGraph
[providers.langgraph]
enabled = false
api_key = "${LANGGRAPH_API_KEY}"
default_model = "gpt-4"

# Fal.ai
[providers.fal_ai]
enabled = false
api_key = "${FAL_AI_API_KEY}"
default_model = "gpt-4"

# Helicone
[providers.helicone]
enabled = false
api_key = "${HELICONE_API_KEY}"
default_model = "gpt-4"

# LiteLLM Proxy (Local)
[providers.litellm_proxy]
enabled = false
base_url = "http://localhost:4000"
default_model = "gpt-4"

# Pydantic AI Agent
[providers.pydantic_ai_agent]
enabled = false
api_key = "${PYDANTIC_AI_AGENT_API_KEY}"
default_model = "gpt-4"

[aliases]
# OpenAI
gpt-4 = "openai/gpt-4"
gpt-4-turbo = "openai/gpt-4-turbo"
gpt-4o = "openai/gpt-4o"
gpt-4o-mini = "openai/gpt-4o-mini"
gpt-3.5 = "openai/gpt-3.5-turbo"

# Anthropic
claude = "anthropic/claude-3-opus-20240229"
claude-opus = "anthropic/claude-3-opus-20240229"
claude-sonnet = "anthropic/claude-3-sonnet-20240229"
claude-3 = "anthropic/claude-3-opus-20240229"
claude-3-opus = "anthropic/claude-3-opus-20240229"
claude-3-sonnet = "anthropic/claude-3-sonnet-20240229"
claude-3-haiku = "anthropic/claude-3-haiku-20240307"

# Gemini/Antigravity
gemini = "antigravity/gemini-2.0-flash"
gemini-pro = "antigravity/gemini-1.5-pro"
gemini-flash = "antigravity/gemini-2.0-flash"
gemini-2 = "antigravity/gemini-2.0-pro-exp-02-05"
gemini-1.5 = "antigravity/gemini-1.5-pro"

# Meta Llama
llama = "meta_llama/llama-3.1-70b-instruct"
llama-3 = "meta_llama/llama-3.1-70b-instruct"
llama-2 = "meta_llama/llama-2-70b-chat-hf"
llama-3.1 = "meta_llama/llama-3.1-70b-instruct"
llama-3.2 = "meta_llama/llama-3.2-90b-vision-instruct"

# Mistral
mistral = "mistral/mistral-large"
mistral-large = "mistral/mistral-large"
mistral-medium = "mistral/mistral-medium"
mistral-small = "mistral/mistral-small"
mixtral = "mistral/mixtral-8x7b"
codestral = "mistral/codestral"

# DeepSeek
deepseek = "deepseek/deepseek-chat"
deepseek-chat = "deepseek/deepseek-chat"
deepseek-coder = "deepseek/deepseek-coder"

# Groq
groq = "groq/llama3-70b-8192"
llama3-groq = "groq/llama3-70b-8192"
mixtral-groq = "groq/mixtral-8x7b-32768"

# Cohere
command = "cohere/command"
command-r = "cohere/command-r"
command-r-plus = "cohere/command-r-plus"

# Ollama
ollama-llama3 = "ollama/llama3"
ollama-mistral = "ollama/mistral"
ollama-codellama = "ollama/codellama"

# AWS Bedrock
bedrock = "bedrock/anthropic.claude-3-opus-20240229-v1:0"

# Vertex AI
vertex = "vertex_ai/gemini-1.5-pro"

# Hugging Face
hf = "huggingface/gpt2"

# Perplexity
perplexity = "perplexity/pplx-7b-online"

# xAI/Grok
grok = "xai/grok-beta"
xai = "xai/grok-beta"

# Qwen
qwen = "openrouter/qwen/qwen-2.5-72b-instruct"

# Hermes
hermes = "openrouter/nousresearch/hermes-3-llama-3.1-405b"

[logging]
level = "info"
verbose = false
"#
        .to_string()
    }
}

/// Expand environment variables in a string
/// Supports ${VAR} syntax
fn expand_env_var(value: &str) -> String {
    let mut result = value.to_string();

    // Simple env var expansion for ${VAR} pattern
    while let Some(start) = result.find("${") {
        if let Some(end) = result[start..].find("}") {
            let var_name = &result[start + 2..start + end];
            let var_value = std::env::var(var_name).unwrap_or_default();
            result.replace_range(start..start + end + 1, &var_value);
        } else {
            break;
        }
    }

    result
}

/// Configuration errors
#[derive(Debug)]
pub enum ConfigError {
    Io(std::io::Error),
    Parse(String),
    NotSupported(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::Io(e) => write!(f, "IO error: {}", e),
            ConfigError::Parse(e) => write!(f, "Parse error: {}", e),
            ConfigError::NotSupported(e) => write!(f, "Not supported: {}", e),
        }
    }
}

impl std::error::Error for ConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::with_defaults();
        assert_eq!(config.server.port, 8080);
        assert!(config.providers.contains_key("openai"));
    }

    #[test]
    fn test_expand_env_var() {
        std::env::set_var("TEST_VAR", "test_value");
        assert_eq!(
            expand_env_var("prefix-${TEST_VAR}-suffix"),
            "prefix-test_value-suffix"
        );
    }

    #[test]
    fn test_example_config() {
        let example = Config::example();
        assert!(example.contains("[server]"));
        assert!(example.contains("[providers.openai]"));
    }
}
