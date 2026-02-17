# LLMG — LLM Gateway

A high-performance Rust LLM gateway and provider library. One OpenAI-compatible API for **70+ LLM providers**.

[![CI](https://github.com/modpotatodotdev/LLMG/actions/workflows/ci.yml/badge.svg)](https://github.com/modpotatodotdev/LLMG/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)

**[Documentation](https://modpotatodotdev.github.io/LLMG)**

## Features

- **Unified API** — Single OpenAI-compatible endpoint for every provider
- **70+ Providers** — OpenAI, Anthropic, Azure, Groq, Mistral, Cohere, DeepSeek, Ollama, OpenRouter, and many more
- **Library + Gateway** — Use as a Rust crate or deploy the HTTP gateway
- **Feature-Gated** — Compile only the providers you need
- **Streaming** — Server-Sent Events across all providers
- **Rig Integration** — Drop-in provider for [Rig](https://github.com/0xPlaygrounds/rig) agents

## Quick Start

### Gateway

```bash
cargo build --release --bin llmg-gateway
OPENAI_API_KEY=sk-... ./target/release/llmg-gateway
```

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "openai/gpt-4", "messages": [{"role": "user", "content": "Hello!"}]}'
```

### Docker

```bash
docker pull ghcr.io/modpotatodotdev/llmg:latest
docker run -p 8080:8080 -e OPENAI_API_KEY=sk-... ghcr.io/modpotatodotdev/llmg:latest
```

### Library

```toml
[dependencies]
llmg-core = "0.1"
llmg-providers = { version = "0.1", features = ["openai"] }
```

```rust
use llmg_providers::openai::OpenAiClient;
use llmg_core::provider::Provider;
use llmg_core::types::{ChatCompletionRequest, Message};

let client = OpenAiClient::from_env()?;
let request = ChatCompletionRequest {
    model: "gpt-4".to_string(),
    messages: vec![Message::User { content: "Hello!".to_string(), name: None }],
    ..Default::default()
};
let response = client.chat_completion(request).await?;
```

## How Routing Works

Requests use the `provider/model` format:

```
openai/gpt-4              → OpenAI
anthropic/claude-3-opus   → Anthropic
groq/llama3-70b-8192      → Groq
ollama/llama3             → Ollama (local)
openrouter/openai/gpt-4   → OpenRouter (nested)
```

Built-in aliases let you use short names like `gpt-4`, `claude`, or `gemini`.

## Project Structure

| Crate | Purpose |
|-------|---------|
| `llmg-core` | Shared types, traits, error handling |
| `llmg-providers` | Provider implementations (feature-gated) |
| `llmg-gateway` | HTTP gateway server (Axum) |

## Configuration

Set API keys as environment variables. The gateway auto-registers providers based on which keys are present.

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
```

See the [documentation](https://modpotatodotdev.github.io/LLMG/providers/all/) for the full list of providers and their environment variables.

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT) at your option.

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
