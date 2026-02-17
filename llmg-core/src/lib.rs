//! # LLMG Core
//!
//! Core types, traits, and error handling for the LLMG (LLM Gateway) ecosystem.
//!
//! This crate provides the [`Provider`](provider::Provider) trait that all LLM provider
//! implementations must satisfy, along with OpenAI-compatible request/response types.
//!
//! ## Crate Structure
//!
//! - [`types`] — OpenAI-compatible request and response types
//! - [`provider`] — The `Provider` trait, `ProviderRegistry`, and authentication
//! - [`client`] — HTTP client wrapper and builder
//! - [`error`] — Error response types
//! - [`rig`] — Integration with the Rig agent framework
//! - [`streaming`] — SSE streaming chunk types

pub mod client;
pub mod error;
pub mod provider;
pub mod rig;
pub mod streaming;
pub mod types;

pub use client::*;
pub use error::*;
pub use provider::*;
pub use rig::*;
pub use streaming::*;
pub use types::*;
