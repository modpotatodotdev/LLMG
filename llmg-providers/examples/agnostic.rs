use llmg_core::provider::{ProviderRegistry, RoutingProvider};
// Note: RigAdapter would be used here if `rig-core` compatibility is desired,
// but for verify script we just check basic wiring.
use llmg_core::provider::Provider;
use llmg_providers::utils::register_all_from_env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Create registry
    let mut registry = ProviderRegistry::new();

    // 2. Auto-register providers from env
    // This will check features and env vars
    register_all_from_env(&mut registry).await;

    println!("Registered providers: {:?}", registry.list());

    // 3. Create RoutingProvider
    let router = RoutingProvider::new(registry);

    // 4. Verify it implements Provider
    println!("Router provider name: {}", router.provider_name());

    // 5. Verify it supports models from underlying providers
    // (This might be empty if no env vars set, but code path is exercised)
    let models = router.supported_models();
    println!("Supported models: {:?}", models);

    Ok(())
}
