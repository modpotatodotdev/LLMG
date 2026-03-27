# Changelog

## [Unreleased]

### Added
- **Responses API**: OpenAI-compatible Responses API at `/v1/responses` (REST + WebSocket)
  - Create responses with text input and custom function tools
  - SSE streaming and WebSocket streaming
  - Multi-turn conversations via `previous_response_id`
  - Response storage, retrieval, deletion
  - Cancel, compact, count_tokens, submit_tool_outputs, list_input_items endpoints
- **Response store**: In-memory response persistence with `ResponseStore`
- **Streaming tool calls**: `ToolCallDelta` and `ToolCallFunctionDelta` types for streaming tool invocations
- **Streaming usage stats**: Optional `usage` field on `ChatCompletionChunk`
- Audit reports for gateway proxy and OpenAI Responses API coverage

### Changed
- `GatewayState` now includes `response_store` field
- `GatewayState` wrapped in `Arc` for shared WebSocket/REST handler access
- `ChatCompletionChunk` and `DeltaContent` updated with new optional fields
- Provider parsers (OpenAI, Anthropic, Ollama, Z.ai) updated for streaming type changes
- Gateway dependencies: added `ws` feature on axum, `tokio-tungstenite`, `base64`
