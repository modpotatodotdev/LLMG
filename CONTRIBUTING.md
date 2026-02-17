# Contributing to LLMG

Thanks for your interest in contributing to LLMG! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/modpotatodotdev/LLMG.git
cd LLMG
cargo build
cargo test
```

## Project Structure

| Crate | Purpose |
|-------|---------|
| `llmg-core` | Shared types, `Provider` trait, error handling |
| `llmg-providers` | Provider implementations (one file per provider) |
| `llmg-gateway` | HTTP gateway server |
| `docs/` | Astro Starlight documentation site |

## Adding a Provider

1. Create `llmg-providers/src/your_provider.rs`
2. Implement the `Provider` trait from `llmg-core`
3. Add `pub mod your_provider;` to `llmg-providers/src/lib.rs`
4. Register the provider in `llmg-gateway/src/providers.rs`
5. Add documentation in `docs/src/content/docs/providers/`
6. Add tests for client creation and request/response conversion

Every provider must implement:
- `chat_completion()` — Chat completions
- `embeddings()` — Text embeddings (or return `UnsupportedFeature`)
- `provider_name()` — Unique provider identifier
- `from_env()` — Create from environment variables

## Code Style

- Run `cargo fmt` before committing
- Run `cargo clippy -- -D warnings` and fix all warnings
- Follow existing patterns in the codebase

## Pull Requests

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run `cargo fmt && cargo clippy -- -D warnings && cargo test`
5. Open a PR against `main`

## License

By contributing, you agree that your contributions will be licensed under the MIT OR Apache-2.0 license.
