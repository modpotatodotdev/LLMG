//! Core types for LLMG - LLM Gateway
//!
//! These types are compatible with the OpenAI API specification.

use serde::{Deserialize, Serialize};

/// Specifies the format that the model must output
///
/// Used to enforce structured outputs like JSON or JSON Schema validation.
/// Compatible with OpenAI, Anthropic (via prompt engineering), and other providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    /// Enables JSON mode, ensuring the model outputs valid JSON
    JsonObject,
    /// Enables structured outputs with JSON Schema validation
    JsonSchema {
        /// The JSON Schema configuration
        json_schema: JsonSchemaConfig,
    },
    /// Plain text output (default behavior)
    Text,
}

/// Configuration for JSON Schema structured outputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonSchemaConfig {
    /// The name of the response format (used for identification)
    pub name: String,
    /// The JSON Schema object describing the output structure
    pub schema: serde_json::Value,
    /// Whether to enforce strict schema adherence (supported by OpenAI)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
    /// A description of what the schema represents
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl JsonSchemaConfig {
    /// Create a new JSON Schema configuration
    pub fn new(name: impl Into<String>, schema: serde_json::Value) -> Self {
        Self {
            name: name.into(),
            schema,
            strict: None,
            description: None,
        }
    }

    /// Enable strict mode for strict schema adherence
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = Some(strict);
        self
    }

    /// Add a description for the schema
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

impl ResponseFormat {
    /// Create a JSON Schema response format from a schema configuration
    pub fn json_schema(config: JsonSchemaConfig) -> Self {
        Self::JsonSchema {
            json_schema: config,
        }
    }

    /// Check if this is a structured output format (JSON Object or JSON Schema)
    pub fn is_structured(&self) -> bool {
        matches!(self, Self::JsonObject | Self::JsonSchema { .. })
    }

    /// Get the JSON schema if this is a JSON Schema format
    pub fn schema(&self) -> Option<&JsonSchemaConfig> {
        match self {
            Self::JsonSchema { json_schema } => Some(json_schema),
            _ => None,
        }
    }
}

/// Request body for chat completion requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    /// ID of the model to use
    pub model: String,

    /// A list of messages comprising the conversation
    pub messages: Vec<Message>,

    /// What sampling temperature to use (0-2)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// The maximum number of tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Whether to stream back partial progress
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// An alternative to sampling with temperature
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Number between -2.0 and 2.0 for frequency penalty
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    /// Number between -2.0 and 2.0 for presence penalty
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    /// Up to 4 sequences where the API will stop generating
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,

    /// A unique identifier representing your end-user
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// A list of tools the model may call
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,

    /// Controls which (if any) tool is called by the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Specifies the format that the model must output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
}

/// A tool that can be called by the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// The type of the tool. Currently, only "function" is supported.
    pub r#type: String,
    /// The function definition
    pub function: FunctionDefinition,
}

/// A function definition for a tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    /// The name of the function to be called
    pub name: String,
    /// A description of what the function does
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The parameters the functions accepts, described as a JSON Schema object.
    pub parameters: serde_json::Value,
}

/// Controls which tool is called by the model
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// "none" means the model will not call a function
    /// "auto" means the model can pick between generating a message or calling a function
    /// "required" means the model must call one or more tools
    String(String),
    /// Specifies a specific tool to call
    Named(NamedToolChoice),
}

/// A specific tool to be called by the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedToolChoice {
    /// The type of the tool. Currently, only "function" is supported.
    pub r#type: String,
    /// The function to call
    pub function: FunctionName,
}

/// A function name identifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionName {
    /// The name of the function to call
    pub name: String,
}

/// A message in the conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    /// A system message
    System {
        /// The content of the message
        content: String,
        /// An optional name for the participant
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// A user message
    User {
        /// The content of the message
        content: String,
        /// An optional name for the participant
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// An assistant message
    Assistant {
        /// The content of the message
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        /// The refusal message if the model refused to respond
        #[serde(default, skip_serializing_if = "Option::is_none")]
        refusal: Option<String>,
        /// The tool calls generated by the model
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<ToolCall>>,
    },
    /// A tool message
    Tool {
        /// The content of the message
        content: String,
        /// Tool call that this message is responding to
        tool_call_id: String,
    },
}

/// A tool call generated by the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// The ID of the tool call
    pub id: String,
    /// The type of the tool. Currently, only "function" is supported.
    pub r#type: String,
    /// The function that the model called
    pub function: FunctionCall,
}

/// A function call generated by the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    /// The name of the function to call
    pub name: String,
    /// The arguments to call the function with, as generated by the model in JSON format.
    pub arguments: String,
}

/// Response from a chat completion request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    /// A unique identifier for the completion
    pub id: String,

    /// The object type, which is always "chat.completion"
    pub object: String,

    /// The Unix timestamp when the completion was created
    pub created: i64,

    /// The model used for completion
    pub model: String,

    /// A list of completion choices
    pub choices: Vec<Choice>,

    /// Usage statistics for the completion request
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// A completion choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    /// The index of the choice
    pub index: u32,
    /// A chat completion message generated by the model
    pub message: Message,
    /// The reason the completion finished
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Usage statistics for a completion request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    /// Number of tokens in the prompt
    pub prompt_tokens: u32,
    /// Number of tokens in the completion
    pub completion_tokens: u32,
    /// Total number of tokens used
    pub total_tokens: u32,
}

/// Request body for embedding requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    /// ID of the model to use
    pub model: String,
    /// Input text to embed
    pub input: String,
    /// The format to return the embeddings in
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
    /// Dimensions of the output embedding
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    /// A unique identifier representing your end-user
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

/// Response from an embedding request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// A unique identifier for the embedding
    pub id: String,
    /// The object type, which is always "list"
    pub object: String,
    /// The list of embeddings
    pub data: Vec<Embedding>,
    /// The model used for the embedding
    pub model: String,
    /// Usage statistics for the request
    pub usage: Usage,
}

/// A single embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    /// The index of the embedding in the list
    pub index: u32,
    /// The object type, which is always "embedding"
    pub object: String,
    /// The embedding vector
    pub embedding: Vec<f32>,
}
