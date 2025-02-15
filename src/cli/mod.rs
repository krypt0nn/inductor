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

        #[arg(long, default_value_t = 1024 * 1024 * 64)]
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

        #[arg(long, default_value_t = 1024 * 1024 * 16)]
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

        #[arg(long, default_value_t = 1024 * 1024 * 64)]
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

        #[arg(long, short, default_value_t = 128)]
        /// Amount of dimensions in a word embedding.
        embedding_size: usize,

        #[arg(long, short, default_value_t = 4)]
        /// Amount of tokens used to predict the next one.
        context_tokens_num: usize,

        #[arg(long, short, default_value_t = 5000)]
        /// Amount of tokens after which position encoding will start repeating.
        ///
        /// If set to 0 no positional encoding is applied.
        position_encoding_period: usize,

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

            Self::TextGenerator {
                model,
                embedding_size,
                context_tokens_num,
                position_encoding_period,
                command
            } => {
                command.execute(
                    model,
                    embedding_size,
                    context_tokens_num,
                    position_encoding_period
                )
            }

            Self::Serve { port } => {
                let device = WgpuDevice::default();

                println!("🚀 Hosting your GPU under {}", format!("ws://0.0.0.0:{port}").yellow());

                burn::server::start::<Wgpu>(device, port);

                Ok(())
            }
        }
    }
}
