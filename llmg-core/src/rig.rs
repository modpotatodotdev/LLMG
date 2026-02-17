//! Rig framework integration for LLMG
//!
//! Allows LLMG providers to be used with the Rig agentic AI framework.
//! Enable with the "rig" feature flag.

use crate::{
    provider::{LlmError, Provider},
    types::{ChatCompletionRequest, ChatCompletionResponse, Message as LlmMessage},
};

/// Adapter for using LLMG providers with Rig
pub struct RigAdapter<P: Provider> {
    provider: P,
    model: String,
}

impl<P: Provider + Clone> RigAdapter<P> {
    /// Create a new Rig adapter
    pub fn new(provider: P, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
        }
    }

    /// Create a completion request builder
    pub fn completion(&self) -> RigCompletionBuilder<P> {
        RigCompletionBuilder::new(self.provider.clone(), self.model.clone())
    }
}

/// Builder for completion requests
pub struct RigCompletionBuilder<P: Provider> {
    provider: P,
    model: String,
    messages: Vec<LlmMessage>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
}

impl<P: Provider> RigCompletionBuilder<P> {
    fn new(provider: P, model: String) -> Self {
        Self {
            provider,
            model,
            messages: Vec::new(),
            temperature: None,
            max_tokens: None,
        }
    }

    /// Add a system message
    pub fn system(mut self, content: impl Into<String>) -> Self {
        self.messages.push(LlmMessage::System {
            content: content.into(),
            name: None,
        });
        self
    }

    /// Add a user message
    pub fn user(mut self, content: impl Into<String>) -> Self {
        self.messages.push(LlmMessage::User {
            content: content.into(),
            name: None,
        });
        self
    }

    /// Set temperature
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set max tokens
    pub fn max_tokens(mut self, max: u32) -> Self {
        self.max_tokens = Some(max);
        self
    }

    /// Send the completion request
    pub async fn send(self) -> Result<RigCompletion, LlmError> {
        let request = ChatCompletionRequest {
            model: self.model,
            messages: self.messages,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            stream: Some(false),
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            user: None,
            tools: None,
            tool_choice: None,
        };

        let response = self.provider.chat_completion(request).await?;

        Ok(RigCompletion::from(response))
    }
}

/// Completion response wrapper for Rig
pub struct RigCompletion {
    pub content: String,
    pub model: String,
    pub usage: Option<crate::types::Usage>,
}

impl From<ChatCompletionResponse> for RigCompletion {
    fn from(response: ChatCompletionResponse) -> Self {
        let content = response
            .choices
            .first()
            .and_then(|choice| match &choice.message {
                LlmMessage::Assistant { content, .. } => content.clone(),
                _ => None,
            })
            .unwrap_or_default();

        Self {
            content,
            model: response.model,
            usage: response.usage,
        }
    }
}

impl std::fmt::Display for RigCompletion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.content)
    }
}

/// Example integration helper
///
/// ```rust
/// use llmg_core::provider::Provider;
/// use llmg_providers::openai::OpenAiClient;
///
/// async fn example() -> Result<(), Box<dyn std::error::Error>> {
///     // Create LLMG provider
///     let openai = OpenAiClient::from_env()?;
///     
///     // Wrap in Rig adapter
///     let adapter = llmg_core::rig::RigAdapter::new(openai, "gpt-4");
///     
///     // Use with Rig-style API
///     let completion = adapter
///         .completion()
///         .system("You are a helpful assistant")
///         .user("Hello!")
///         .send()
///         .await?;
///     
///     println!("Response: {}", completion);
///     Ok(())
/// }
/// ```
#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{LlmError, Provider};
    use crate::types::{
        ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse,
    };

    #[derive(Clone, Debug)]
    struct MockProvider;

    #[async_trait::async_trait]
    impl Provider for MockProvider {
        async fn chat_completion(
            &self,
            _request: ChatCompletionRequest,
        ) -> Result<ChatCompletionResponse, LlmError> {
            Ok(ChatCompletionResponse {
                id: "test".to_string(),
                object: "chat.completion".to_string(),
                created: 0,
                model: "test-model".to_string(),
                choices: vec![crate::types::Choice {
                    index: 0,
                    message: LlmMessage::Assistant {
                        content: Some("Test response".to_string()),
                        refusal: None,
                        tool_calls: None,
                    },
                    finish_reason: Some("stop".to_string()),
                }],
                usage: None,
            })
        }

        async fn embeddings(
            &self,
            _request: EmbeddingRequest,
        ) -> Result<EmbeddingResponse, LlmError> {
            unimplemented!()
        }

        fn provider_name(&self) -> &'static str {
            "mock"
        }
    }

    #[tokio::test]
    async fn test_rig_adapter() {
        let adapter = RigAdapter::new(MockProvider, "test-model");
        let completion = adapter
            .completion()
            .system("Test system")
            .user("Test user")
            .send()
            .await;

        assert!(completion.is_ok());
        assert_eq!(completion.unwrap().content, "Test response");
    }
}
