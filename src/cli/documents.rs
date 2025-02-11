use std::path::PathBuf;

use clap::Parser;
use colorful::Colorful;

use crate::prelude::*;

#[derive(serde::Deserialize)]
struct DiscordChat {
    pub guild: DiscordGuild,
    pub channel: DiscordChannel,
    pub messages: Vec<DiscordMessage>
}

#[derive(serde::Deserialize)]
struct DiscordGuild {
    pub name: String
}

#[derive(serde::Deserialize)]
struct DiscordChannel {
    // pub category: String,
    pub name: String,
    pub topic: Option<String>
}

#[derive(serde::Deserialize)]
struct DiscordMessage {
    pub content: String,
    pub author: DiscordAuthor
}

#[derive(serde::Deserialize)]
struct DiscordAuthor {
    pub name: String,
    // pub nickname: String
}

#[derive(Parser)]
pub enum DocumentsCLI {
    /// Create new documents database.
    Create,

    /// Insert document into the database.
    Insert {
        #[arg(long)]
        /// Path to the document file.
        document: PathBuf,

        #[arg(long)]
        /// Convert content of the document to lowercase.
        lowercase: bool,

        #[arg(long)]
        /// Read input file as discord chat history export in JSON format.
        discord_chat: bool,

        #[arg(long)]
        /// Split discord chat messages into separate documents.
        discord_split_documents: bool,

        #[arg(long, default_value_t = 0)]
        /// Use last N messages from the discord chat history.
        ///
        /// When 0 is set (default), then all messages are used.
        discord_last_n: usize
    }
}

impl DocumentsCLI {
    #[inline]
    pub fn execute(self, database: PathBuf, cache_size: i64) -> anyhow::Result<()> {
        match self {
            Self::Create => {
                let database = database.canonicalize().unwrap_or(database);

                println!("⏳ Creating documents database in {database:?}...");

                match DocumentsDatabase::open(&database, cache_size) {
                    Ok(_) => {
                        println!("{}", "🚀 Database created".green());
                        println!("📖 {} {} command will create new database automatically if needed", "Note:".blue(), "`documents insert`".yellow());
                    }

                    Err(err) => eprintln!("{}", format!("🧯 Failed to create database: {err}").red())
                }
            }

            Self::Insert { document, lowercase, discord_chat, discord_split_documents, discord_last_n } => {
                let database = database.canonicalize().unwrap_or(database);

                println!("⏳ Opening documents database in {database:?}...");

                let database = match DocumentsDatabase::open(&database, cache_size) {
                    Ok(database) => database,
                    Err(err) => {
                        eprintln!("{}", format!("🧯 Failed to open database: {err}").red());

                        return Ok(());
                    }
                };

                let document = document.canonicalize().unwrap_or(document);

                println!("⏳ Reading document {document:?}...");

                match std::fs::read_to_string(document) {
                    Ok(mut document) if discord_chat => {
                        if lowercase {
                            document = document.to_lowercase();
                        }

                        let chat = match serde_json::from_str::<DiscordChat>(&document) {
                            Ok(chat) => chat,
                            Err(err) => {
                                eprintln!("{}", format!("🧯 Failed to parse chat history: {err}").red());

                                return Ok(());
                            }
                        };

                        drop(document);

                        let chat_name = format!(
                            "<server>{}</server><channel>#{}</channel><topic>{}</topic>",
                            &chat.guild.name,
                            &chat.channel.name,
                            chat.channel.topic.as_deref().unwrap_or("")
                        );

                        let messages = if discord_last_n == 0 || discord_last_n >= chat.messages.len() {
                            &chat.messages
                        } else {
                            &chat.messages[chat.messages.len() - discord_last_n..]
                        };

                        println!("⏳ Inserting {} chat messages...", messages.len());

                        if discord_split_documents {
                            for i in 0..messages.len() {
                                let message = &messages[i];

                                let prev_message = messages.get(i - 1)
                                    .map(|message| message.content.as_str())
                                    .unwrap_or("");

                                let document = Document::default()
                                    .with_input(prev_message)
                                    .with_context(format!("{chat_name}<author>@{}</author>", message.author.name))
                                    .with_output(&message.content);

                                if let Err(err) = database.insert(&document) {
                                    eprintln!("{}", format!("🧯 Failed to insert document: {err}").red());

                                    return Ok(());
                                }
                            }
                        }

                        else {
                            let document = messages.iter()
                                .map(|message| format!(
                                    "<message>@{}: {}</message>",
                                    message.author.name,
                                    message.content
                                ))
                                .fold(String::new(), |acc, message| acc + &message);

                            let document = Document::new(document)
                                .with_context(&chat_name);

                            if let Err(err) = database.insert(&document) {
                                eprintln!("{}", format!("🧯 Failed to insert document: {err}").red());

                                return Ok(());
                            }
                        }

                        println!("{}", "✅ Documents inserted".green());
                    }

                    Ok(mut document) => {
                        if lowercase {
                            document = document.to_lowercase();
                        }

                        let document = Document::new(document);

                        println!("⏳ Inserting document...");

                        match database.insert(&document) {
                            Ok(_) => println!("{}", "✅ Document inserted".green()),
                            Err(err) => eprintln!("{}", format!("🧯 Failed to insert document: {err}").red())
                        }
                    }

                    Err(err) => eprintln!("{}", format!("🧯 Failed to read document file: {err}").red())
                }
            }
        }

        Ok(())
    }
}
