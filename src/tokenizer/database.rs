use std::path::Path;

use rusqlite::Connection;

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub id: i64,
    pub value: String,
    pub occurences: i64,
    pub frequency: f64
}

#[derive(Debug)]
pub struct DatabaseInsertsTransaction<'connection>(rusqlite::Transaction<'connection>);

impl DatabaseInsertsTransaction<'_> {
    /// Add token insertion to the transaction.
    pub fn insert_token(&self, token: impl AsRef<str>) -> rusqlite::Result<()> {
        self.0.prepare_cached("UPDATE tokens SET occurences = occurences + 1 WHERE value = ?1;")?
            .execute([token.as_ref()])?;

        self.0.prepare_cached("INSERT OR IGNORE INTO tokens (value, occurences) VALUES (?1, 1);")?
            .execute([token.as_ref()])?;

        Ok(())
    }

    /// Commit transaction and update tokens' frequencies.
    pub fn commit(self) -> rusqlite::Result<()> {
        self.0.execute_batch("UPDATE tokens SET frequency = CAST(occurences AS REAL) / (SELECT SUM(occurences) FROM tokens);")?;

        self.0.commit()
    }
}

#[derive(Debug)]
/// SQLite database for storing word tokens.
pub struct Database {
    connection: Connection
}

impl Database {
    /// Open database with given cache size.
    /// Negative number means sqlite pages (1024 bytes), positive - bytes.
    pub fn open(path: impl AsRef<Path>, cache_size: i64) -> rusqlite::Result<Self> {
        let connection = Connection::open(path)?;

        connection.execute_batch(&format!("
            PRAGMA cache_size = {cache_size};

            PRAGMA journal_mode = MEMORY;
            PRAGMA temp_store = MEMORY;
            PRAGMA synchronous = OFF;

            CREATE TABLE IF NOT EXISTS tokens (
                id          INTEGER          NOT NULL,
                value       TEXT     UNIQUE  NOT NULL,
                occurences  INTEGER          NOT NULL  DEFAULT 0,
                frequency   REAL             NOT NULL  DEFAULT 0.0,

                PRIMARY KEY (id)
            );

            CREATE INDEX IF NOT EXISTS idx_tokens_value ON tokens (value);

            INSERT OR IGNORE INTO tokens (id, value, occurences, frequency) VALUES (0, '', 0, 0.0);
        "))?;

        Ok(Self {
            connection
        })
    }

    /// Query token from the database using its id.
    ///
    /// Guaranteed to return `Ok(None)` if word is not stored.
    pub fn query_token_by_id(&self, token: i64) -> anyhow::Result<Option<Token>> {
        let result = self.connection.prepare_cached("SELECT value, occurences, frequency FROM tokens WHERE id = ?1")?
            .query_row([token], |row| {
                Ok(Token {
                    id: token,
                    value: row.get(0)?,
                    occurences: row.get(1)?,
                    frequency: row.get(2)?
                })
            });

        match result {
            Ok(token) => Ok(Some(token)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(err) => anyhow::bail!(err)
        }
    }

    /// Query token from the database using its original word.
    ///
    /// Guaranteed to return `Ok(None)` if token is not stored.
    pub fn query_token_by_value(&self, token: impl AsRef<str>) -> anyhow::Result<Option<Token>> {
        let result = self.connection.prepare_cached("SELECT id, occurences, frequency FROM tokens WHERE value = ?1")?
            .query_row([token.as_ref()], |row| {
                Ok(Token {
                    id: row.get(0)?,
                    value: token.as_ref().to_string(),
                    occurences: row.get(1)?,
                    frequency: row.get(2)?
                })
            });

        match result {
            Ok(token) => Ok(Some(token)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(err) => anyhow::bail!(err)
        }
    }

    /// Open tokens insertion transaction.
    ///
    /// After committing this transaction will calculate frequencies
    /// of each token in the database.
    pub fn insert_tokens(&mut self) -> rusqlite::Result<DatabaseInsertsTransaction<'_>> {
        Ok(DatabaseInsertsTransaction(self.connection.transaction()?))
    }

    /// Iterate over all tokens stored in the database.
    ///
    /// Return amount of read tokens.
    pub fn for_each(&self, mut callback: impl FnMut(Token) -> anyhow::Result<()>) -> anyhow::Result<u64> {
        let mut tokens = 0;

        self.connection.prepare_cached("SELECT id, value, occurences, frequency FROM tokens ORDER BY id ASC")?
            .query_map([], |row| {
                Ok(Token {
                    id: row.get(0)?,
                    value: row.get(1)?,
                    occurences: row.get(2)?,
                    frequency: row.get(3)?
                })
            })?
            .try_for_each(|token| {
                tokens += 1;

                callback(token?)
            })?;

        Ok(tokens)
    }
}

#[test]
fn test_tokens_database() -> anyhow::Result<()> {
    let _ = std::fs::remove_file("tokens_database.db");

    let mut db = Database::open("tokens_database.db", 4096)?;

    assert!(db.query_token_by_value("hello")?.is_none());

    let transaction = db.insert_tokens()?;

    transaction.insert_token("hello")?;
    transaction.insert_token("world")?;

    transaction.commit()?;

    assert_eq!(db.query_token_by_value("hello")?.unwrap().id, 1);
    assert_eq!(db.query_token_by_value("world")?.unwrap().id, 2);

    assert_eq!(db.query_token_by_value("hello")?.unwrap().frequency, 0.5);
    assert_eq!(db.query_token_by_value("world")?.unwrap().frequency, 0.5);

    let _ = std::fs::remove_file("tokens_database.db");

    Ok(())
}
