use llmg_core::provider::ProviderRegistry;
use std::sync::Arc;

/// Automatically register all enabled providers that have valid configuration in the environment.
///
/// This function checks for the presence of feature flags and corresponding environment variables
/// for each supported provider. If a provider is enabled and configured, it is added to the registry.
pub async fn register_all_from_env(registry: &mut ProviderRegistry) {
    // --- Tier 1: OpenAI-compatible (simple API key) ---
    #[cfg(feature = "openai")]
    {
        if let Ok(client) = crate::openai::OpenAiClient::from_env() {
            registry.register(Arc::new(client));
        }
    }

    #[cfg(feature = "anthropic")]
    {
        if let Ok(client) = crate::anthropic::AnthropicClient::from_env() {
            registry.register(Arc::new(client));
        }
    }

    #[cfg(feature = "groq")]
    {
        if let Ok(client) = crate::groq::GroqClient::from_env() {
            registry.register(Arc::new(client));
        }
    }

    #[cfg(feature = "mistral")]
    {
        if let Ok(client) = crate::mistral::MistralClient::from_env() {
            registry.register(Arc::new(client));
        }
    }

    #[cfg(feature = "cohere")]
    {
        if let Ok(client) = crate::cohere::CohereClient::from_env() {
            registry.register(Arc::new(client));
        }
    }

    #[cfg(feature = "deepseek")]
    {
        if let Ok(client) = crate::deepseek::DeepseekClient::from_env() {
            registry.register(Arc::new(client));
        }
    }

    #[cfg(feature = "openrouter")]
    {
        if let Ok(client) = crate::openrouter::OpenRouterClient::from_env() {
            registry.register(Arc::new(client));
        }
    }

    #[cfg(feature = "perplexity")]
    {
        if let Ok(client) = crate::perplexity::PerplexityClient::from_env() {
            registry.register(Arc::new(client));
        }
    }

    #[cfg(feature = "antigravity")]
    {
        match crate::antigravity::AntigravityClient::new().await {
            Ok(client) => {
                registry.register(Arc::new(client));
            }
            Err(e) => {
                tracing::debug!("Antigravity not initialized: {:?}", e);
            }
        }
    }

    #[cfg(feature = "together_ai")]
    {
        if let Ok(client) = crate::together_ai::TogetherAiClient::from_env() {
            registry.register(Arc::new(client));
        }
    }

    #[cfg(feature = "fireworks_ai")]
    {
        if let Ok(client) = crate::fireworks_ai::FireworksAiClient::from_env() {
            registry.register(Arc::new(client));
        }
    }

    #[cfg(feature = "deepinfra")]
    {
        if let Ok(client) = crate::deepinfra::DeepInfraClient::from_env() {
            registry.register(Arc::new(client));
        }
    }

    #[cfg(feature = "anyscale")]
    {
        if let Ok(client) = crate::anyscale::AnyscaleClient::from_env() {
            registry.register(Arc::new(client));
        }
    }

    // --- Tier 2: Specialized / Enterprise ---

    #[cfg(feature = "azure")]
    {
        if let Ok(client) = crate::azure::AzureOpenAiClient::from_env() {
            registry.register(Arc::new(client));
        }
    }

    #[cfg(feature = "vertex_ai")]
    {
        if let Ok(client) = crate::vertex_ai::VertexAiClient::from_env() {
            registry.register(Arc::new(client));
        }
    }

    #[cfg(feature = "bedrock")]
    {
        if let Ok(client) = crate::bedrock::BedrockClient::from_env() {
            registry.register(Arc::new(client));
        }
    }

    #[cfg(feature = "xai")]
    {
        if let Ok(client) = crate::xai::XaiClient::from_env() {
            registry.register(Arc::new(client));
        }
    }

    #[cfg(feature = "github_copilot")]
    {
        match crate::github_copilot::GitHubCopilotClient::new().await {
            Ok(client) => {
                registry.register(Arc::new(client));
            }
            Err(e) => {
                tracing::debug!("GitHub Copilot not initialized (likely missing tokens or feature disabled): {:?}", e);
            }
        }
    }

    // --- Tier 3: Local / Self-Hosted ---

    #[cfg(feature = "ollama")]
    {
        let client = crate::ollama::OllamaClient::from_env();
        registry.register(Arc::new(client));
    }

    #[cfg(feature = "vllm")]
    {
        let client = crate::vllm::VllmClient::from_env();
        registry.register(Arc::new(client));
    }

    #[cfg(feature = "lm_studio")]
    {
        let client = crate::lm_studio::LmStudioClient::from_env();
        registry.register(Arc::new(client));
    }

    #[cfg(feature = "z_ai")]
    {
        if let Ok(client) = crate::z_ai::ZaiClient::from_env() {
            registry.register(Arc::new(client));
        }
        if let Ok(client) = crate::z_ai::ZaiClient::coding_from_env() {
            registry.register(Arc::new(client));
        }
    }
}
