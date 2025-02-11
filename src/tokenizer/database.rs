use std::path::Path;
use std::sync::{Arc, Mutex};

use rusqlite::Connection;

use crate::prelude::*;

#[derive(Debug, Clone)]
/// SQLite database for storing word tokens.
pub struct Database {
    connection: Arc<Mutex<Connection>>
}

impl Database {
    /// Open database with given cache size.
    /// Negative number means sqlite pages (1024 bytes), positive - bytes.
    pub fn open(path: impl AsRef<Path>, cache_size: i64) -> rusqlite::Result<Self> {
        let connection = Connection::open(path)?;

        connection.execute(&format!("PRAGMA cache_size = {cache_size};"), ())?;

        connection.execute_batch(&format!("
            CREATE TABLE IF NOT EXISTS tokens (
                id    INTEGER NOT NULL,
                value TEXT UNIQUE NOT NULL,

                PRIMARY KEY (id),
                CONSTRAINT id_max CHECK(id < {EMBEDDING_MAX_TOKENS})
            );

            CREATE INDEX IF NOT EXISTS idx_tokens_value on tokens (value);

            INSERT OR IGNORE INTO tokens (id, value) VALUES (0, '');
        "))?;

        Ok(Self {
            connection: Arc::new(Mutex::new(connection))
        })
    }

    /// Query token from the database.
    ///
    /// Guaranteed to return `Ok(None)` if token is not stored.
    pub fn query_token(&self, token: impl AsRef<str>) -> anyhow::Result<Option<i64>> {
        let connection = self.connection.lock()
            .map_err(|_| anyhow::anyhow!("Failed to lock sqlite connection"))?;

        let id = connection.prepare_cached("SELECT id FROM tokens WHERE value = ?1")?
            .query_row([token.as_ref()], |row| row.get::<_, i64>(0));

        match id {
            Ok(id) => Ok(Some(id)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(err) => anyhow::bail!(err)
        }
    }

    /// Insert token to the database.
    ///
    /// Return id of inserted token.
    pub fn insert_token(&self, token: impl AsRef<str>) -> anyhow::Result<i64> {
        let Some(id) = self.query_token(token.as_ref())? else {
            let connection = self.connection.lock()
                .map_err(|_| anyhow::anyhow!("Failed to lock sqlite connection"))?;

            connection.prepare_cached("INSERT INTO tokens (value) VALUES (?1)")?
                .execute([token.as_ref()])?;

            return Ok(connection.last_insert_rowid());
        };

        Ok(id)
    }

    /// Query word from the database.
    ///
    /// Guaranteed to return `Ok(None)` if word is not stored.
    pub fn query_word(&self, token: i64) -> anyhow::Result<Option<String>> {
        let connection = self.connection.lock()
            .map_err(|_| anyhow::anyhow!("Failed to lock sqlite connection"))?;

        let word = connection.prepare_cached("SELECT value FROM tokens WHERE id = ?1")?
            .query_row([token], |row| row.get::<_, String>(0));

        match word {
            Ok(word) => Ok(Some(word)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(err) => anyhow::bail!(err)
        }
    }

    /// Iterate over all tokens stored in the database.
    ///
    /// Return amount of read tokens.
    pub fn for_each(&self, mut callback: impl FnMut(i64, String) -> anyhow::Result<()>) -> anyhow::Result<u64> {
        let mut tokens = 0;

        let connection = self.connection.lock()
            .map_err(|_| anyhow::anyhow!("Failed to lock sqlite connection"))?;

        connection.prepare_cached("SELECT id, value FROM tokens ORDER BY id ASC")?
            .query_map([], move |row| {
                let id = row.get::<_, i64>(0)?;
                let token = row.get::<_, String>(1)?;

                Ok((id, token))
            })?
            .try_for_each(|row| {
                let (id, token) = row?;

                tokens += 1;

                callback(id, token)
            })?;

        Ok(tokens)
    }
}

#[test]
fn test_tokens_database() -> anyhow::Result<()> {
    let _ = std::fs::remove_file("tokens_database.db");

    let db = Database::open("tokens_database.db", 4096)?;

    assert!(db.query_token("hello")?.is_none());

    db.insert_token("hello")?;
    db.insert_token("world")?;

    assert_eq!(db.query_token("hello")?, Some(1));
    assert_eq!(db.query_token("world")?, Some(2));

    let _ = std::fs::remove_file("tokens_database.db");

    Ok(())
}
