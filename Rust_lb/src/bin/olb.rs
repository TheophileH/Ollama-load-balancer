use clap::{Parser, Subcommand};
use reqwest::Client;
use scraper::{Html, Selector};
use std::process::Command;
use inquire::Select;
use fuzzy_matcher::FuzzyMatcher;
use fuzzy_matcher::skim::SkimMatcherV2;
use std::fmt;

#[derive(Parser)]
#[command(name = "olb")]
#[command(about = "Ollama Load Balancer CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Discover models from ollama.com
    Discover {
        /// Search query
        query: String,
    },
    /// Pull a model
    Pull {
        /// Model name
        model_id: String,
    },
    /// List local models
    List,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Discover { query } => {
            cmd_discover(query).await;
        }
        Commands::Pull { model_id } => {
            cmd_pull(model_id);
        }
        Commands::List => {
            cmd_list();
        }
    }
}

fn cmd_pull(model: &str) {
    println!("⬇️  Pulling {} via ollama...", model);
    let status = Command::new("ollama")
        .arg("pull")
        .arg(model)
        .status();

    match status {
        Ok(s) => {
            if s.success() {
                println!("✅ Successfully pulled {}", model);
            } else {
                eprintln!("❌ Failed to pull {}", model);
            }
        }
        Err(e) => eprintln!("❌ Failed to execute ollama pull: {}", e),
    }
}

fn cmd_list() {
    let _ = Command::new("ollama").arg("list").status();
}

// Struct to hold item data for Inquire
#[derive(Clone)]
struct ModelItem {
    index: usize,
    raw_name: String, // For fuzzy matching
    display: String,  // For rendering
    value: String,    // For returning (the model path/cmd)
}

impl fmt::Display for ModelItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display)
    }
}

async fn cmd_discover(query: &str) {
    loop {
        let url = format!("https://ollama.com/search?q={}", query);
        println!("🔍 Searching https://ollama.com for '{}'...", query);

        let client = Client::new();
        let resp = match client.get(&url).send().await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("❌ Failed to search: {}", e);
                return;
            }
        };

        let text = match resp.text().await {
            Ok(t) => t,
            Err(e) => {
                eprintln!("❌ Failed to get response text: {}", e);
                return;
            }
        };

        let document = Html::parse_document(&text);
        let a_selector = Selector::parse("a").unwrap();
        
        let re_pulls = regex::Regex::new(r"([\d\.]+[KMB]?)\s*Pulls").unwrap();
        let re_tags = regex::Regex::new(r"(\d+)\s*Tags").unwrap();
        let re_updated = regex::Regex::new(r"Updated\s+(.+)").unwrap();
        
        // Regex for valid model paths: /namespace/model (excluding reserved words)
        let re_model_path = regex::Regex::new(r"^/([^/]+)/([^/]+)$").unwrap();
        let reserved = ["blog", "search", "download", "docs", "cloud", "public", "cdn-cgi", "login", "signup", "legal", "terms", "privacy"];

        let mut models_data = Vec::new();

        for element in document.select(&a_selector) {
            if let Some(href) = element.value().attr("href") {
                if let Some(caps) = re_model_path.captures(href) {
                     let namespace = &caps[1];
                     // Filter out reserved paths
                     if !reserved.contains(&namespace) {
                         let model_path = href.trim_start_matches('/').to_string(); // e.g., "library/llama3" or "user/repo"
                         
                         // Avoid duplicates
                         if !models_data.iter().any(|(m,_,_,_,_)| m == &model_path) {
                            let full_text = element.text().collect::<Vec<_>>().join(" ");
                            
                            let pulls = re_pulls.captures(&full_text)
                                .map(|c| c.get(1).map_or("?", |m| m.as_str()))
                                .unwrap_or("?");
                                
                            let tags = re_tags.captures(&full_text)
                                .map(|c| c.get(1).map_or("?", |m| m.as_str()))
                                .unwrap_or("?");

                            let updated = re_updated.captures(&full_text)
                                .map(|c| c.get(1).map_or("?", |m| m.as_str()))
                                .unwrap_or("?");

                            // Display Name: strip "library/" for cleanliness, keep others
                            let display_name = if model_path.starts_with("library/") {
                                model_path.trim_start_matches("library/").to_string()
                            } else {
                                model_path.clone()
                            };

                            models_data.push((model_path, display_name, pulls.to_string(), tags.to_string(), updated.to_string()));
                         }
                     }
                }
            }
        }

        if models_data.is_empty() {
            println!("❌ No models found for '{}'", query);
            return;
        }

        println!("Found {} models:", models_data.len());

        // Calculate max width for alignment
        let max_len = models_data.iter().map(|(_, name, _, _, _)| name.len()).max().unwrap_or(30);
        let idx_width = models_data.len().to_string().len();

        let items: Vec<ModelItem> = models_data.iter().enumerate().map(|(i, (path, name, pulls, tags, updated))| {
            let display = format!("[{:>w$}] - {:<width$}  (⬇️ {:<5} | 🏷️ {:<3} | 🕒 {})", 
                i + 1, name, pulls, tags, updated, w = idx_width, width = max_len);
            
            ModelItem {
                index: i + 1,
                raw_name: name.clone(),
                display,
                value: path.clone(),
            }
        }).collect();

        // Custom filter function for Inquire
        let matcher = SkimMatcherV2::default();
        let filter = move |input: &str, item: &ModelItem, _string_value: &str, _index: usize| -> bool {
            let input = input.trim();
            // 1. Exact Number Match
            if let Ok(num) = input.parse::<usize>() {
                return item.index == num;
            }
            // 2. Fuzzy Name Match
            if matcher.fuzzy_match(&item.raw_name, input).is_some() {
                return true;
            }
            false
        };

        let selection = Select::new("Select a model family (Type number or name)", items)
            .with_filter(&filter)
            .with_page_size(15)
            .prompt();

        match selection {
            Ok(item) => {
                 if select_model_tag(&client, &item.value).await {
                    break;
                }
            },
            Err(_) => {
                println!("Bye!");
                break;
            }
        }
    }
}

// Returns true if action taken, false if "Go Back" requested
async fn select_model_tag(client: &Client, model_path: &str) -> bool {
    let url = format!("https://ollama.com/{}/tags", model_path);
    let display_name = if model_path.starts_with("library/") {
        model_path.trim_start_matches("library/")
    } else {
        model_path
    };
    
    println!("🔍 Fetching all tags for '{}'...", display_name);

    let resp = match client.get(&url).send().await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("❌ Failed to fetch model details: {}", e);
            return true;
        }
    };

    let text = match resp.text().await {
        Ok(t) => t,
        Err(e) => {
            eprintln!("❌ Failed to get text: {}", e);
            return true;
        }
    };

    let document = Html::parse_document(&text);

    let a_selector = Selector::parse("a").unwrap();
    
    struct TagInfo {
        name: String,
        size: String,
        context: String,
        input: String,
        pull_cmd: String,
    }
    
    let mut tags = Vec::new();

    let prefix = format!("/{}:", model_path);
    
    // Regexes for scraping
    let re_size = regex::Regex::new(r"([\d\.]+[GM]B)").unwrap();
    let re_context = regex::Regex::new(r"(\d+[KM])\s*(?:context window)?").unwrap();
    let re_input = regex::Regex::new(r"((?:Text(?:, Image)?|Image))").unwrap();

    for element in document.select(&a_selector) {
        if let Some(href) = element.value().attr("href") {
            if href.starts_with(&prefix) {
                // e.g. /library/llama3:latest or /user/repo:tag
                let pull_cmd = href.trim_start_matches('/').to_string();
                
                // For display name, we just want the tag part
                let tag_part = pull_cmd.trim_start_matches(model_path).trim_start_matches(':').to_string();

                // Get all text content
                let raw_text = element.text().collect::<Vec<_>>().join(" ");
                
                // Coalesce all whitespace (including newlines) into single spaces
                let re_whitespace = regex::Regex::new(r"\s+").unwrap();
                let full_text = re_whitespace.replace_all(&raw_text, " ").to_string();

                // Check if this looks like a valid tag row (has size info)
                if full_text.contains("GB") || full_text.contains("MB") {
                     // Extract metadata using regex
                     let size = re_size.captures(&full_text)
                        .map(|c| c[1].to_string())
                        .unwrap_or("?".to_string());
                     
                     let context = re_context.captures(&full_text)
                        .map(|c| c[1].to_string())
                        .unwrap_or("?".to_string());
                     
                     let input = re_input.captures(&full_text)
                        .map(|c| c[1].to_string())
                        .unwrap_or("?".to_string());
                     
                     tags.push(TagInfo {
                         name: tag_part, // Display just the tag
                         size,
                         context,
                         input,
                         pull_cmd, // Full identifier for pulling
                     });
                }
            }
        }
    }
    
    tags.dedup_by(|a, b| a.name == b.name);
    
    if tags.is_empty() {
        println!("⚠️  No detailed tags found using page scraper. Falling back to 'latest'.");
        cmd_pull(&format!("{}:latest", model_path));
        return true;
    }
    
    // Calculate max width
    let max_len = tags.iter().map(|t| t.name.len()).max().unwrap_or(20);
    // Index width calculation requires acknowledging the extra "Go Back" item
    let total_items = tags.len() + 1;
    let idx_width = total_items.to_string().len();

    let mut items: Vec<ModelItem> = Vec::new();
    
    // Add Go Back
    items.push(ModelItem {
        index: 0,
        raw_name: "back".to_string(),
        display: format!("[{:>w$}] - ⬅️  Go Back to Model Families", 0, w = idx_width),
        value: "BACK".to_string(),
    });

    for (i, t) in tags.iter().enumerate() {
        let display = format!("[{:>w$}] - {:<width$} | 💾 {:<8} | 🧠 {:<8} | ⌨️  {:<10}", 
            i + 1, t.name, t.size, t.context, t.input, w = idx_width, width = max_len);
            
        items.push(ModelItem {
            index: i + 1,
            raw_name: t.name.clone(),
            display,
            value: t.pull_cmd.clone(),
        });
    }

    let matcher = SkimMatcherV2::default();
    let filter = move |input: &str, item: &ModelItem, _string_value: &str, _index: usize| -> bool {
        let input = input.trim();
        // 1. Exact Number Match
        if let Ok(num) = input.parse::<usize>() {
            return item.index == num;
        }
        // 2. Fuzzy Name Match
        if matcher.fuzzy_match(&item.raw_name, input).is_some() {
            return true;
        }
        false
    };

    let selection = Select::new(
        &format!("Select variant for {} (Type number or tag)", display_name),
        items,
    )
    .with_filter(&filter)
    .with_page_size(15)
    .prompt();

    match selection {
        Ok(item) => {
            if item.value == "BACK" {
                return false;
            }
            println!("🎯 Selected: {}", item.value);
            cmd_pull(&item.value);
            true
        },
        Err(_) => {
            println!("Bye!");
            true
        }
    }
}
