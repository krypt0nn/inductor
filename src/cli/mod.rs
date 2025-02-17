use std::path::PathBuf;

use clap::Parser;
use colorful::Colorful;

use burn::backend::{Wgpu, wgpu::WgpuDevice};

use crate::prelude::*;

pub mod config;
pub mod documents;
pub mod tokens;
pub mod embeddings;
pub mod text_generator;

#[derive(Parser)]
pub struct Cli {
    #[arg(long, short)]
    /// Path to the project's config file.
    pub config: Option<PathBuf>,

    #[command(subcommand)]
    pub command: CliVariant
}

#[derive(Parser)]
pub enum CliVariant {
    /// Create new project.
    Init,

    /// Manage datasets of plain text corpuses.
    Documents {
        #[command(subcommand)]
        command: documents::DocumentsCli
    },

    /// Manage datasets of plain text tokens.
    Tokens {
        #[command(subcommand)]
        command: tokens::TokensCli
    },

    /// Manage embeddings of plain text tokens.
    Embeddings {
        #[command(subcommand)]
        command: embeddings::EmbeddingsCli
    },

    /// Manage text generation model.
    TextGenerator {
        #[command(subcommand)]
        command: text_generator::TextGeneratorCli
    },

    /// Host your device for remote computations.
    Serve {
        #[arg(long, short, default_value_t = 3000)]
        port: u16
    }
}

impl Cli {
    #[inline]
    pub fn execute(self) -> anyhow::Result<()> {
        let config_path = self.config.unwrap_or_else(|| PathBuf::from("inductor.toml"));
        let config_path = config_path.canonicalize().unwrap_or(config_path);

        let config = config::load(&config_path).unwrap_or_default();

        match self.command {
            CliVariant::Init => {
                println!("⏳ Saving config file in {config_path:?}...");

                if let Err(err) = std::fs::write(&config_path, toml::to_string_pretty(&config)?) {
                    eprintln!("{}", format!("🧯 Failed to save config file in {config_path:?}: {err}").red());
                }

                // Load it again so now default relative paths are made absolute.
                let config = config::load(&config_path).unwrap_or_default();

                println!("⏳ Initializing documents database in {:?}...", config.documents.database_path);

                if let Err(err) = DocumentsDatabase::open(&config.documents.database_path, config.documents.ram_cache) {
                    eprintln!("{}", format!("🧯 Failed to open documents database: {err}").red());

                    return Ok(());
                }

                println!("⏳ Initializing tokens database in {:?}...", config.tokens.database_path);

                if let Err(err) = TokensDatabase::open(&config.tokens.database_path, config.tokens.ram_cache) {
                    eprintln!("{}", format!("🧯 Failed to open tokens database: {err}").red());

                    return Ok(());
                }

                println!("⏳ Initializing word embeddings database in {:?}...", config.embeddings.database_path);

                if let Err(err) = WordEmbeddingsDatabase::open(&config.embeddings.database_path, config.embeddings.ram_cache) {
                    eprintln!("{}", format!("🧯 Failed to open word embeddings database: {err}").red());

                    return Ok(());
                }

                println!("{}", "🚀 Project created".green());

                Ok(())
            }

            CliVariant::Documents { command } => command.execute(config),
            CliVariant::Tokens { command } => command.execute(config),
            CliVariant::Embeddings { command } => command.execute(config),
            CliVariant::TextGenerator { command } => command.execute(config),

            CliVariant::Serve { port } => {
                let device = WgpuDevice::default();

                println!("🚀 Hosting your GPU under {}", format!("ws://0.0.0.0:{port}").yellow());

                burn::server::start::<Wgpu>(device, port);

                Ok(())
            }
        }
    }
}
