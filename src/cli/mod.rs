use std::path::PathBuf;

use clap::Parser;

pub mod documents;
pub mod tokens;
pub mod embeddings;

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
    }
}

impl CLI {
    #[inline]
    pub fn execute(self) -> anyhow::Result<()> {
        match self {
            Self::Documents { database, cache_size, command } => command.execute(database, cache_size),
            Self::Tokens { database, cache_size, command } => command.execute(database, cache_size),
            Self::Embeddings { database, cache_size, command } => command.execute(database, cache_size)
        }
    }
}
