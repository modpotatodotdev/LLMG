use futures::StreamExt;
use llmg_core::provider::Provider;
use llmg_core::types::{ChatCompletionRequest, Message};
use llmg_providers::github_copilot::GitHubCopilotClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = GitHubCopilotClient::new().await?;

    let req = ChatCompletionRequest {
        model: "gpt-4o".to_string(),
        messages: vec![Message::User {
            content: "Count to 10 slowly. Put a new line after each number.".to_string(),
            name: None,
        }],
        temperature: Some(0.7),
        max_tokens: Some(100),
        stream: Some(true),
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        stop: None,
        user: None,
        tools: None,
        tool_choice: None,
    };

    println!("Sending request...");
    let mut stream = client.chat_completion_stream(req).await?;

    println!("Stream started. Waiting for chunks...");
    while let Some(chunk_res) = stream.next().await {
        match chunk_res {
            Ok(chunk) => {
                if let Some(choice) = chunk.choices.first() {
                    if let Some(text) = &choice.delta.content {
                        print!("{}", text);
                    }
                }
            }
            Err(e) => {
                eprintln!("Stream error: {:?}", e);
                break;
            }
        }
    }
    println!("\nStream finished.");

    Ok(())
}
