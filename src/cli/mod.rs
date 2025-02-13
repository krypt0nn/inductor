use std::path::PathBuf;

use clap::Parser;
use colorful::Colorful;

use burn::backend::{Wgpu, wgpu::WgpuDevice};

pub mod documents;
pub mod tokens;
pub mod embeddings;
pub mod text_generator;

#[derive(Parser)]
pub enum CLI {
    /// Manage datasets of plain text corpuses.
    Documents {
        #[arg(long, short)]
        /// Path to the database file.
        database: PathBuf,

        #[arg(long, default_value_t = 1024 * 1024 * 32)]
        /// SQLite database cache size.
        ///
        /// Positive value sets cache size in bytes, negative - in sqlite pages.
        cache_size: i64,

        #[command(subcommand)]
        command: documents::DocumentsCLI
    },

    /// Manage datasets of plain text tokens.
    Tokens {
        #[arg(long, short)]
        /// Path to the database file.
        database: PathBuf,

        #[arg(long, default_value_t = 1024 * 1024)]
        /// SQLite database cache size.
        ///
        /// Positive value sets cache size in bytes, negative - in sqlite pages.
        cache_size: i64,

        #[command(subcommand)]
        command: tokens::TokensCLI
    },

    /// Manage embeddings of plain text tokens.
    Embeddings {
        #[arg(long, short)]
        /// Path to the database file.
        database: PathBuf,

        #[arg(long, default_value_t = 1024 * 1024 * 32)]
        /// SQLite database cache size.
        ///
        /// Positive value sets cache size in bytes, negative - in sqlite pages.
        cache_size: i64,

        #[command(subcommand)]
        command: embeddings::EmbeddingsCLI
    },

    /// Manage text generation model.
    TextGenerator {
        #[arg(long, short)]
        /// Path to the text generation model.
        model: PathBuf,

        #[command(subcommand)]
        command: text_generator::TextGeneratorCLI
    },

    /// Host your device for remote computations.
    Serve {
        #[arg(long, short, default_value_t = 3000)]
        port: u16
    }
}

impl CLI {
    #[inline]
    pub fn execute(self) -> anyhow::Result<()> {
        match self {
            Self::Documents { database, cache_size, command } => command.execute(database, cache_size),
            Self::Tokens { database, cache_size, command } => command.execute(database, cache_size),
            Self::Embeddings { database, cache_size, command } => command.execute(database, cache_size),
            Self::TextGenerator { model, command } => command.execute(model),

            Self::Serve { port } => {
                let device = WgpuDevice::default();

                println!("🚀 Hosting your GPU under {}", format!("ws://0.0.0.0:{port}").yellow());

                burn::server::start::<Wgpu>(device, port);

                Ok(())
            }
        }
    }
}
