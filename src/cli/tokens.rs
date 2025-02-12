use std::path::PathBuf;
use std::collections::HashSet;

use clap::Parser;
use colorful::Colorful;

use crate::prelude::*;

#[derive(Parser)]
pub enum TokensCLI {
    /// Create new tokens database.
    Create,

    /// Insert all tokens from the documents database.
    Update {
        #[arg(long, short)]
        /// Path to the documents database.
        documents: PathBuf,

        #[arg(long, short)]
        /// Convert content of the documents to lowercase.
        lowercase: bool,

        #[arg(long, short)]
        /// Strip punctuation from the documents.
        strip_punctuation: bool
    }
}

impl TokensCLI {
    #[inline]
    pub fn execute(self, database: PathBuf, cache_size: i64) -> anyhow::Result<()> {
        match self {
            Self::Create => {
                let database = database.canonicalize().unwrap_or(database);

                println!("⏳ Creating tokens database in {database:?}...");

                match TokensDatabase::open(&database, cache_size) {
                    Ok(_) => {
                        println!("{}", "🚀 Database created".green());
                        println!("📖 {} {} command will create new database automatically if needed", "Note:".blue(), "`tokens update`".yellow());
                    }

                    Err(err) => eprintln!("{}", format!("🧯 Failed to create database: {err}").red())
                }
            }

            Self::Update { documents, lowercase, strip_punctuation } => {
                let tokens = database.canonicalize().unwrap_or(database);
                let documents = documents.canonicalize().unwrap_or(documents);

                println!("⏳ Opening tokens database in {tokens:?}...");

                let tokens = match TokensDatabase::open(&tokens, cache_size) {
                    Ok(tokens) => tokens,
                    Err(err) => {
                        eprintln!("{}", format!("🧯 Failed to open tokens database: {err}").red());

                        return Ok(());
                    }
                };

                println!("⏳ Opening documents database in {documents:?}...");

                let documents = match DocumentsDatabase::open(&documents, cache_size) {
                    Ok(documents) => documents,
                    Err(err) => {
                        eprintln!("{}", format!("🧯 Failed to open documents database: {err}").red());

                        return Ok(());
                    }
                };

                let parser = DocumentsParser::new(lowercase, strip_punctuation);

                documents.for_each(|document| {
                    // Make new one for each document so this struct potentially won't fill whole RAM.
                    let mut inserted_tokens = HashSet::new();

                    for token in parser.read_document(document) {
                        if inserted_tokens.insert(token.clone()) {
                            tokens.insert_token(token)?;
                        }
                    }

                    Ok(())
                })?;
            }
        }

        Ok(())
    }
}
