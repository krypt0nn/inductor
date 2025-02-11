use clap::Parser;
use colorful::Colorful;

pub mod documents;
pub mod tokenizer;
pub mod embeddings;
pub mod cli;

pub mod prelude {
    pub use super::documents::prelude::*;
    pub use super::tokenizer::prelude::*;
    pub use super::embeddings::prelude::*;
}

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    if let Err(err) = cli::CLI::parse().execute() {
        eprintln!("{}", format!("🧯 An error occured: {err}").red());
    }
}
