use axum::{
    extract::Json,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use llmg_core::types::ChatCompletionRequest;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::signal;

mod cache;
mod config;
mod middleware;
mod providers;
mod response_store;
mod responses_rest;
mod routing;
mod streaming;
mod websocket;

use cache::LlmCache;
use config::Config;
use llmg_core::provider::ProviderRegistry;
use providers::create_registry;
use response_store::ResponseStore;
use routing::{route_chat_completion, route_chat_completion_stream};
use axum::extract::{State, ws::WebSocketUpgrade};

pub struct GatewayState {
    pub config: Config,
    pub registry: ProviderRegistry,
    pub cache: LlmCache,
    pub response_store: ResponseStore,
}

impl GatewayState {
    pub async fn new(config: Config) -> Self {
        let registry = create_registry(&config).await;
        Self {
            config,
            registry,
            cache: LlmCache::new(),
            response_store: ResponseStore::new(),
        }
    }
}

/// Health check endpoint
async fn health_check() -> (StatusCode, &'static str) {
    (StatusCode::OK, "OK")
}

/// List available models
async fn list_models(
    axum::extract::State(state): axum::extract::State<std::sync::Arc<GatewayState>>,
) -> (StatusCode, axum::Json<serde_json::Value>) {
    let providers: Vec<serde_json::Value> = state
        .registry
        .list()
        .into_iter()
        .map(|name| {
            serde_json::json!({
                "id": name,
                "object": "model",
                "owned_by": name,
            })
        })
        .collect();

    let models = serde_json::json!({
        "object": "list",
        "data": providers
    });
    (StatusCode::OK, axum::Json(models))
}

/// Chat completions endpoint
async fn chat_completions(
    axum::extract::State(state): axum::extract::State<std::sync::Arc<GatewayState>>,
    Json(mut request): Json<ChatCompletionRequest>,
) -> Response {
    // Resolve any aliases in the model name
    request.model = state
        .config
        .aliases
        .get(&request.model)
        .cloned()
        .unwrap_or(request.model);

    if request.stream == Some(true) {
        return match route_chat_completion_stream(state, request).await {
            Ok(response) => response,
            Err(err) => err.into_response(),
        };
    }

    if let Some(cached_response) = state.cache.get(&request).await {
        return (StatusCode::OK, Json(cached_response)).into_response();
    }

    match route_chat_completion(state.clone(), request).await {
        Ok(response) => response,
        Err(err) => err.into_response(),
    }
}

async fn responses_ws(
    State(state): State<Arc<GatewayState>>,
    ws: WebSocketUpgrade,
) -> Response {
    websocket::responses_handler(ws, State(state)).await
}

async fn responses_rest(
    State(state): State<Arc<GatewayState>>,
    axum::extract::Query(query): axum::extract::Query<responses_rest::StreamQuery>,
    Json(request): Json<responses_rest::CreateResponseRequest>,
) -> Response {
    responses_rest::create_response_handler(State(state), axum::extract::Query(query), Json(request)).await
}

async fn retrieve_response(
    State(state): State<Arc<GatewayState>>,
    axum::extract::Path(response_id): axum::extract::Path<String>,
) -> Response {
    responses_rest::retrieve_response_handler(State(state), axum::extract::Path(response_id)).await
}

async fn delete_response(
    State(state): State<Arc<GatewayState>>,
    axum::extract::Path(response_id): axum::extract::Path<String>,
) -> Response {
    responses_rest::delete_response_handler(State(state), axum::extract::Path(response_id)).await
}

async fn cancel_response(
    State(state): State<Arc<GatewayState>>,
    axum::extract::Path(response_id): axum::extract::Path<String>,
) -> Response {
    responses_rest::cancel_response_handler(State(state), axum::extract::Path(response_id)).await
}

async fn list_input_items(
    State(state): State<Arc<GatewayState>>,
    axum::extract::Path(response_id): axum::extract::Path<String>,
) -> Response {
    responses_rest::list_input_items_handler(State(state), axum::extract::Path(response_id)).await
}

async fn compact_response(
    State(state): State<Arc<GatewayState>>,
    axum::extract::Path(response_id): axum::extract::Path<String>,
) -> Response {
    responses_rest::compact_response_handler(State(state), axum::extract::Path(response_id)).await
}

async fn count_tokens(
    State(state): State<Arc<GatewayState>>,
    Json(request): Json<responses_rest::CreateResponseRequest>,
) -> Response {
    responses_rest::count_tokens_handler(State(state), Json(request)).await
}

async fn submit_tool_outputs(
    State(state): State<Arc<GatewayState>>,
    axum::extract::Path(response_id): axum::extract::Path<String>,
    axum::extract::Query(query): axum::extract::Query<responses_rest::StreamQuery>,
    Json(request): Json<responses_rest::SubmitToolOutputsRequest>,
) -> Response {
    responses_rest::submit_tool_outputs_handler(State(state), axum::extract::Path(response_id), axum::extract::Query(query), Json(request)).await
}

/// Create the Axum router
pub fn create_app(state: std::sync::Arc<GatewayState>) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/responses/ws", post(responses_ws))
        .route("/v1/responses", post(responses_rest))
        .route("/v1/responses/{response_id}", get(retrieve_response).delete(delete_response))
        .route("/v1/responses/{response_id}/cancel", post(cancel_response))
        .route("/v1/responses/{response_id}/input_items", get(list_input_items))
        .route("/v1/responses/{response_id}/submit_tool_outputs", post(submit_tool_outputs))
        .route("/v1/responses/{response_id}/compact", post(compact_response))
        .route("/v1/responses/count_tokens", post(count_tokens))
        .with_state(state)
        .layer(axum::middleware::from_fn(middleware::rate_limit_middleware))
        .layer(axum::middleware::from_fn(middleware::auth_middleware))
        .layer(axum::middleware::from_fn(middleware::timeout_middleware))
        .layer(axum::middleware::from_fn(middleware::request_id_middleware))
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .layer(middleware::create_cors_layer())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("Shutdown signal received, starting graceful shutdown...");
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load configuration
    let config = Config::load_from_dir(".").unwrap_or_else(|e| {
        eprintln!("Warning: Failed to load config, using defaults: {}", e);
        Config::with_defaults()
    });

    // Initialize rate limiter
    middleware::init_rate_limiter(config.rate_limit.clone());

    // Initialize logging
    if config.logging.verbose {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(config.logging.level.clone())
            .init();
    }

    let port = config.server.port;
    let addr = SocketAddr::from(([0, 0, 0, 0], port));

    let state = std::sync::Arc::new(GatewayState::new(config).await);

    // Spawn cache eviction task
    let state_clone = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(60));
        loop {
            interval.tick().await;
            state_clone.cache.clear_expired().await;
        }
    });

    let app = create_app(state);

    tracing::info!("LLMG Gateway starting on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.map_err(|e| {
        tracing::error!("Failed to bind to {}: {}", addr, e);
        e
    })?;

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| {
            tracing::error!("Server error: {}", e);
            e
        })?;

    tracing::info!("Gateway shutdown complete.");
    Ok(())
}
