//! CLI argument parsing for LLMG Gateway
//!
//! Uses clap for parsing command-line arguments.

use clap::Parser;
use std::path::PathBuf;

/// LLMG Gateway - LLM Gateway Server
#[derive(Parser, Debug, Clone)]
#[command(name = "llmg-gateway")]
#[command(about = "High-performance LLM Gateway with OpenRouter-style routing")]
#[command(version)]
pub struct Cli {
    /// Port to listen on
    #[arg(short, long, env = "LLMG_PORT", default_value = "8080")]
    pub port: u16,

    /// Host address to bind to
    #[arg(short = 'H', long, env = "LLMG_HOST", default_value = "0.0.0.0")]
    pub host: String,

    /// Path to configuration file
    #[arg(short, long, env = "LLMG_CONFIG")]
    pub config: Option<PathBuf>,

    /// Log level (error, warn, info, debug, trace)
    #[arg(short, long, env = "LLMG_LOG_LEVEL", default_value = "info")]
    pub log_level: String,

    /// Enable CORS for all origins
    #[arg(long, env = "LLMG_CORS")]
    pub cors: bool,

    /// Request timeout in seconds
    #[arg(short, long, env = "LLMG_TIMEOUT", default_value = "60")]
    pub timeout: u64,

    /// Enable request/response logging
    #[arg(long, env = "LLMG_VERBOSE")]
    pub verbose: bool,

    /// Comma-separated list of enabled providers
    #[arg(short = 'P', long, env = "LLMG_PROVIDERS", value_delimiter = ',')]
    pub providers: Vec<String>,
}

impl Cli {
    /// Parse CLI arguments from environment
    pub fn parse_args() -> Self {
        Self::parse()
    }

    /// Get the socket address to bind to
    pub fn bind_address(&self) -> std::net::SocketAddr {
        use std::net::{IpAddr, Ipv4Addr, SocketAddr};

        let ip: IpAddr = self.host.parse().unwrap_or_else(|_| {
            eprintln!("Warning: Invalid host '{}', using 0.0.0.0", self.host);
            IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0))
        });

        SocketAddr::new(ip, self.port)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_defaults() {
        let cli = Cli {
            port: 8080,
            host: "0.0.0.0".to_string(),
            config: None,
            log_level: "info".to_string(),
            cors: false,
            timeout: 60,
            verbose: false,
            providers: vec![],
        };

        assert_eq!(cli.port, 8080);
        assert_eq!(cli.host, "0.0.0.0");
        assert_eq!(cli.log_level, "info");
    }
}
