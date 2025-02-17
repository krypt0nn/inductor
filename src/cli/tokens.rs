use std::collections::HashSet;

use clap::Parser;
use colorful::Colorful;

use crate::prelude::*;

#[derive(Parser)]
pub enum TokensCli {
    /// Insert all tokens from the documents database.
    Update
}

impl TokensCli {
    #[inline]
    pub fn execute(self, config: super::config::CliConfig) -> anyhow::Result<()> {
        match self {
            Self::Update => {
                println!("⏳ Opening documents database in {:?}...", config.documents.database_path);

                let documents = match DocumentsDatabase::open(&config.documents.database_path, config.documents.ram_cache) {
                    Ok(documents) => documents,
                    Err(err) => {
                        eprintln!("{}", format!("🧯 Failed to open documents database: {err}").red());

                        return Ok(());
                    }
                };

                println!("⏳ Opening tokens database in {:?}...", config.tokens.database_path);

                let mut tokens = match TokensDatabase::open(&config.tokens.database_path, config.tokens.ram_cache) {
                    Ok(tokens) => tokens,
                    Err(err) => {
                        eprintln!("{}", format!("🧯 Failed to open tokens database: {err}").red());

                        return Ok(());
                    }
                };

                let parser = DocumentsParser::new(
                    config.tokens.lowercase,
                    config.tokens.strip_punctuation,
                    config.tokens.whitespace_tokens
                );

                println!("⏳ Updating tokens database...");

                let mut inserted_tokens = HashSet::new();

                let transaction = match tokens.insert_tokens() {
                    Ok(transaction) => transaction,
                    Err(err) => {
                        eprintln!("{}", format!("🧯 Failed to open tokens insertion transaction: {err}").red());

                        return Ok(());
                    }
                };

                documents.for_each(|document| {
                    for token in parser.read_document(document) {
                        // We insert tokens even if they were already inserted
                        // because tokens database calculates occurences of tokens
                        // within all text corpuses.
                        transaction.insert_token(&token)?;

                        inserted_tokens.insert(token);
                    }

                    Ok(())
                })?;

                match transaction.commit() {
                    Ok(()) => println!("✅ Updated {} tokens", inserted_tokens.len().to_string().yellow()),
                    Err(err) => eprintln!("{}", format!("🧯 Failed to commit tokens insertion transaction: {err}").red())
                }
            }
        }

        Ok(())
    }
}
