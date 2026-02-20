use crate::config::Config;
use llmg_core::provider::{Provider, ProviderRegistry};
use llmg_providers::*;
use std::sync::Arc;

/// Create a provider registry based on configuration
pub async fn create_registry(config: &Config) -> ProviderRegistry {
    let mut registry = ProviderRegistry::new();

    for (name, provider_cfg) in &config.providers {
        if !provider_cfg.enabled {
            continue;
        }

        let provider: Option<Arc<dyn Provider>> = match name.as_str() {
            // --- Tier 1: OpenAI-compatible (simple API key + optional base_url) ---
            "openai" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = OpenAiClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "groq" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = GroqClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "deepinfra" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = DeepInfraClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "together_ai" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = TogetherAiClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "fireworks_ai" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = FireworksAiClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "anyscale" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = AnyscaleClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "deepseek" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = DeepseekClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "perplexity" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = PerplexityClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "z_ai" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = ZaiClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "z_ai_coding" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = ZaiClient::coding(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "sambanova" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = SambaNovaClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "cerebras" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = CerebrasClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "nscale" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = NscaleClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "xai" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = XaiClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "hyperbolic" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = HyperbolicClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "featherless_ai" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = FeatherlessAiClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "friendliai" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = FriendliaiClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "octoai" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = OctoAiClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "openrouter" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = OpenRouterClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }

            // --- Tier 2: Commercial APIs (API key, sometimes custom headers/formats) ---
            "anthropic" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = AnthropicClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "cohere" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = CohereClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "mistral" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = MistralClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "ai21" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = Ai21Client::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "aiml" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = AimlClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "aleph_alpha" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = AlephAlphaClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "apertis_ai" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = ApertisAiClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "chutes" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = ChutesClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "comet" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = CometClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "compactifai" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = CompactifAiClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "meta_llama" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = MetaLlamaClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "minimax" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = MiniMaxClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "nano_gpt" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = NanoGptClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "poe" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = PoeClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "publicai" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = PublicAiClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "synthetic" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = SyntheticClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "v0" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = V0Client::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "volcano" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = VolcanoClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }

            // --- Tier 2: Specialized API key providers ---
            "jina" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = JinaClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "deepgram" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = DeepgramClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "voyageai" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = VoyageaiClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "infinity" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = InfinityClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "milvus" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = MilvusClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "fal_ai" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = FalAiClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "helicone" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = HeliconeClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "langgraph" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = LangGraphClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "pydantic_ai_agent" => {
                if let Some(key) = &provider_cfg.api_key {
                    let mut client = PydanticAiAgentClient::new(key);
                    if let Some(base_url) = &provider_cfg.base_url {
                        client = client.with_base_url(base_url);
                    }
                    Some(Arc::new(client))
                } else {
                    None
                }
            }
            "firecrawl" => {
                if let Some(key) = &provider_cfg.api_key {
                    Some(Arc::new(FirecrawlClient::new(key)))
                } else {
                    None
                }
            }
            "elevenlabs" => {
                if let Some(key) = &provider_cfg.api_key {
                    Some(Arc::new(ElevenLabsClient::new(key)))
                } else {
                    None
                }
            }
            "stability" => {
                if let Some(key) = &provider_cfg.api_key {
                    Some(Arc::new(StabilityAiClient::new(key)))
                } else {
                    None
                }
            }
            "runway" => {
                if let Some(key) = &provider_cfg.api_key {
                    Some(Arc::new(RunwayClient::new(key)))
                } else {
                    None
                }
            }

            // --- Tier 3: Self-hosted / Local (no auth, configurable base_url) ---
            "ollama" => {
                let base_url = provider_cfg
                    .base_url
                    .clone()
                    .unwrap_or_else(|| "http://localhost:11434/v1".to_string());
                Some(Arc::new(OllamaClient::new().with_base_url(base_url)))
            }
            "vllm" => {
                let mut client = VllmClient::new();
                if let Some(base_url) = &provider_cfg.base_url {
                    client = client.with_base_url(base_url);
                }
                if let Some(key) = &provider_cfg.api_key {
                    client = client.with_api_key(key);
                }
                Some(Arc::new(client))
            }
            "lm_studio" => {
                let mut client = LmStudioClient::new();
                if let Some(base_url) = &provider_cfg.base_url {
                    client = client.with_base_url(base_url);
                }
                if let Some(key) = &provider_cfg.api_key {
                    client = client.with_api_key(key);
                }
                Some(Arc::new(client))
            }
            "llamafile" => {
                let mut client = LlamafileClient::new();
                if let Some(base_url) = &provider_cfg.base_url {
                    client = client.with_base_url(base_url);
                }
                if let Some(key) = &provider_cfg.api_key {
                    client = client.with_api_key(key);
                }
                Some(Arc::new(client))
            }
            "oobabooga" => {
                let mut client = OobaboogaClient::new();
                if let Some(base_url) = &provider_cfg.base_url {
                    client = client.with_base_url(base_url);
                }
                if let Some(key) = &provider_cfg.api_key {
                    client = client.with_api_key(key);
                }
                Some(Arc::new(client))
            }
            "triton" => {
                let mut client = TritonClient::new();
                if let Some(base_url) = &provider_cfg.base_url {
                    client = client.with_base_url(base_url);
                }
                if let Some(key) = &provider_cfg.api_key {
                    client = client.with_api_key(key);
                }
                Some(Arc::new(client))
            }
            "petals" => {
                let mut client = PetalsClient::new();
                if let Some(base_url) = &provider_cfg.base_url {
                    client = client.with_base_url(base_url);
                }
                Some(Arc::new(client))
            }
            "docker_runner" => {
                let mut client = DockerRunnerClient::new();
                if let Some(base_url) = &provider_cfg.base_url {
                    client = client.with_base_url(base_url);
                }
                if let Some(key) = &provider_cfg.api_key {
                    client = client.with_api_key(key);
                }
                Some(Arc::new(client))
            }
            "xinference" => {
                let base_url = provider_cfg
                    .base_url
                    .clone()
                    .unwrap_or_else(|| "http://localhost:9997/v1".to_string());
                Some(Arc::new(XinferenceClient::new(base_url)))
            }
            "custom_llm_server" => {
                let mut client = CustomLlmServerClient::new();
                if let Some(base_url) = &provider_cfg.base_url {
                    client = client.with_base_url(base_url);
                }
                Some(Arc::new(client))
            }

            // --- Tier 4: Cloud / Enterprise (complex auth) ---
            "azure" => {
                if let Some(key) = &provider_cfg.api_key {
                    let endpoint = provider_cfg
                        .base_url
                        .clone()
                        .or_else(|| std::env::var("AZURE_OPENAI_ENDPOINT").ok())
                        .unwrap_or_default();
                    let deployment = provider_cfg
                        .headers
                        .get("deployment")
                        .cloned()
                        .or_else(|| std::env::var("AZURE_OPENAI_DEPLOYMENT").ok())
                        .unwrap_or_default();
                    Some(Arc::new(AzureOpenAiClient::new(key, endpoint, deployment)))
                } else {
                    None
                }
            }
            "azure_ai" => {
                if let Some(key) = &provider_cfg.api_key {
                    let endpoint = provider_cfg
                        .base_url
                        .clone()
                        .or_else(|| std::env::var("AZURE_AI_ENDPOINT").ok())
                        .unwrap_or_else(|| "https://api.azure.microsoft.com".to_string());
                    let project_id = provider_cfg
                        .headers
                        .get("project_id")
                        .cloned()
                        .or_else(|| std::env::var("AZURE_AI_PROJECT_ID").ok());
                    Some(Arc::new(AzureAiClient::new(key, endpoint, project_id)))
                } else {
                    None
                }
            }
            "bedrock" => {
                let access_key = provider_cfg
                    .api_key
                    .clone()
                    .or_else(|| std::env::var("AWS_ACCESS_KEY_ID").ok());
                let secret_key = provider_cfg
                    .headers
                    .get("x-aws-secret")
                    .cloned()
                    .or_else(|| std::env::var("AWS_SECRET_ACCESS_KEY").ok());
                let region = provider_cfg
                    .headers
                    .get("x-aws-region")
                    .cloned()
                    .or_else(|| std::env::var("AWS_REGION").ok())
                    .unwrap_or_else(|| "us-west-2".to_string());
                let session_token = provider_cfg
                    .headers
                    .get("x-aws-session-token")
                    .cloned()
                    .or_else(|| std::env::var("AWS_SESSION_TOKEN").ok());

                if let (Some(ak), Some(sk)) = (access_key, secret_key) {
                    Some(Arc::new(BedrockClient::new(ak, sk, region, session_token)))
                } else {
                    None
                }
            }
            "aws_sagemaker" => {
                let access_key = provider_cfg
                    .api_key
                    .clone()
                    .or_else(|| std::env::var("AWS_ACCESS_KEY_ID").ok());
                let secret_key = provider_cfg
                    .headers
                    .get("x-aws-secret")
                    .cloned()
                    .or_else(|| std::env::var("AWS_SECRET_ACCESS_KEY").ok());
                let endpoint = provider_cfg
                    .base_url
                    .clone()
                    .or_else(|| std::env::var("AWS_SAGEMAKER_ENDPOINT").ok())
                    .unwrap_or_else(|| "https://api.sagemaker.aws.amazon.com".to_string());
                let region = provider_cfg
                    .headers
                    .get("x-aws-region")
                    .cloned()
                    .or_else(|| std::env::var("AWS_REGION").ok())
                    .unwrap_or_else(|| "us-east-1".to_string());
                let model_id = provider_cfg
                    .headers
                    .get("model_id")
                    .cloned()
                    .or_else(|| std::env::var("AWS_SAGEMAKER_MODEL_ID").ok());
                let session_token = provider_cfg
                    .headers
                    .get("x-aws-session-token")
                    .cloned()
                    .or_else(|| std::env::var("AWS_SESSION_TOKEN").ok());

                if let (Some(ak), Some(sk)) = (access_key, secret_key) {
                    Some(Arc::new(AwsSagemakerClient::new(
                        ak,
                        sk,
                        endpoint,
                        &region,
                        model_id,
                        session_token,
                    )))
                } else {
                    None
                }
            }
            "vertex_ai" => {
                let api_key = provider_cfg
                    .api_key
                    .clone()
                    .or_else(|| std::env::var("GOOGLE_API_KEY").ok());
                let project_id = provider_cfg
                    .headers
                    .get("project_id")
                    .cloned()
                    .or_else(|| std::env::var("GOOGLE_CLOUD_PROJECT").ok())
                    .unwrap_or_default();
                let location = provider_cfg
                    .headers
                    .get("location")
                    .cloned()
                    .or_else(|| std::env::var("GOOGLE_CLOUD_LOCATION").ok())
                    .unwrap_or_else(|| "us-central1".to_string());

                Some(Arc::new(VertexAiClient::new(
                    api_key, None, project_id, location,
                )))
            }
            "watsonx" => {
                let api_key = provider_cfg
                    .api_key
                    .clone()
                    .or_else(|| std::env::var("WATSONX_API_KEY").ok())
                    .or_else(|| std::env::var("IBM_CLOUD_API_KEY").ok());
                let base_url = provider_cfg
                    .base_url
                    .clone()
                    .or_else(|| std::env::var("WATSONX_URL").ok())
                    .unwrap_or_else(|| "https://us-south.ml.cloud.ibm.com".to_string());
                let project_id = provider_cfg
                    .headers
                    .get("project_id")
                    .cloned()
                    .or_else(|| std::env::var("WATSONX_PROJECT_ID").ok())
                    .unwrap_or_default();

                if let Some(key) = api_key {
                    Some(Arc::new(WatsonxClient::new(key, base_url, project_id)))
                } else {
                    None
                }
            }
            "heroku" => {
                if let Some(key) = &provider_cfg.api_key {
                    let base_url = provider_cfg
                        .base_url
                        .clone()
                        .or_else(|| std::env::var("HEROKU_BASE_URL").ok())
                        .unwrap_or_else(|| "https://us.inference.heroku.com".to_string());
                    let app_name = provider_cfg
                        .headers
                        .get("app_name")
                        .cloned()
                        .or_else(|| std::env::var("HEROKU_APP_NAME").ok());
                    Some(Arc::new(HerokuClient::new(key, base_url, app_name)))
                } else {
                    None
                }
            }
            "huggingface" => {
                let base_url = provider_cfg
                    .base_url
                    .clone()
                    .or_else(|| std::env::var("HF_BASE_URL").ok())
                    .unwrap_or_else(|| "https://api-inference.huggingface.co".to_string());
                let hf_token = provider_cfg
                    .headers
                    .get("hf_token")
                    .cloned()
                    .or_else(|| std::env::var("HF_TOKEN").ok());
                let api_key = provider_cfg
                    .api_key
                    .clone()
                    .or_else(|| std::env::var("HF_API_KEY").ok())
                    .or_else(|| std::env::var("HUGGINGFACE_API_KEY").ok());

                Some(Arc::new(HuggingFaceClient::new(
                    base_url, hf_token, api_key,
                )))
            }

            // --- Tier 5: Specialized / OAuth providers ---
            "github_copilot" => match GitHubCopilotClient::new().await {
                Ok(client) => Some(Arc::new(client)),
                Err(e) => {
                    tracing::warn!("Failed to initialize GitHub Copilot: {e:#}");
                    None
                }
            },
            "antigravity" => {
                // Antigravity uses async OAuth PKCE flow.
                // Similar to GitHub Copilot, requires async initialization.
                None
            }
            "litellm_proxy" => {
                let base_url = provider_cfg
                    .base_url
                    .clone()
                    .or_else(|| std::env::var("LITELLM_PROXY_URL").ok())
                    .unwrap_or_else(|| "http://localhost:4000".to_string());
                let api_key = provider_cfg
                    .api_key
                    .clone()
                    .or_else(|| std::env::var("LITELLM_PROXY_API_KEY").ok());
                let mut client = LitellmProxyClient::new(api_key, base_url);
                if let Some(base_url) = &provider_cfg.base_url {
                    client = client.with_base_url(base_url);
                }
                Some(Arc::new(client))
            }
            "polly" => {
                if let (Some(key), Some(secret)) = (
                    &provider_cfg.api_key,
                    provider_cfg.headers.get("x-aws-secret"),
                ) {
                    let region = provider_cfg
                        .headers
                        .get("x-aws-region")
                        .cloned()
                        .unwrap_or_else(|| "us-east-1".to_string());
                    Some(Arc::new(PollyClient::new(
                        key.clone(),
                        secret.to_string(),
                        region,
                    )))
                } else {
                    None
                }
            }

            _ => None,
        };

        if let Some(p) = provider {
            registry.register(p);
        }
    }

    registry
}
