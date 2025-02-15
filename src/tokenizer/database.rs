use std::path::Path;
use std::rc::Rc;

use rusqlite::Connection;

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub id: i64,
    pub value: String,
    pub occurences: i64,
    pub frequency: Option<f64>
}

#[derive(Debug, Clone)]
/// SQLite database for storing word tokens.
pub struct Database {
    connection: Rc<Connection>
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

                PRIMARY KEY (id)
            );

            CREATE INDEX IF NOT EXISTS idx_tokens_value ON tokens (value);

            CREATE VIEW IF NOT EXISTS tokens_view AS
            SELECT
                id AS token_id,
                value AS token_value,
                occurences AS current_occurences,
                total_occurences,
                (CAST(occurences AS REAL) / CAST(total_occurences AS REAL)) AS frequency
            FROM (
                SELECT id, value, occurences, (SELECT SUM(occurences) FROM tokens) AS total_occurences
                FROM tokens
            );

            INSERT OR IGNORE INTO tokens (id, value, occurences) VALUES (0, '', 0);
        "))?;

        Ok(Self {
            connection: Rc::new(connection)
        })
    }

    /// Query token from the database using its id.
    ///
    /// Guaranteed to return `Ok(None)` if word is not stored.
    pub fn query_token_by_id(&self, token: i64, query_frequency: bool) -> anyhow::Result<Option<Token>> {
        let result = self.connection.prepare_cached("SELECT value, occurences FROM tokens WHERE id = ?1")?
            .query_row([token], |row| {
                let value = row.get::<_, String>(0)?;
                let occurences = row.get::<_, i64>(1)?;

                Ok((value, occurences))
            });

        match result {
            Ok((value, occurences)) => {
                let frequency = query_frequency.then(|| -> rusqlite::Result<f64> {
                    self.connection.prepare_cached("SELECT frequency FROM tokens_view WHERE token_id = ?1")?
                        .query_row([&token], |row| row.get::<_, f64>(0))
                }).transpose()?;

                Ok(Some(Token {
                    id: token,
                    value,
                    occurences,
                    frequency
                }))
            },

            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(err) => anyhow::bail!(err)
        }
    }

    /// Query token from the database using its original word.
    ///
    /// Guaranteed to return `Ok(None)` if token is not stored.
    pub fn query_token_by_value(&self, token: impl ToString, query_frequency: bool) -> anyhow::Result<Option<Token>> {
        let token = token.to_string();

        let result = self.connection.prepare_cached("SELECT id, occurences FROM tokens WHERE value = ?1")?
            .query_row([&token], |row| {
                let id = row.get::<_, i64>(0)?;
                let occurences = row.get::<_, i64>(1)?;

                Ok((id, occurences))
            });

        match result {
            Ok((id, occurences)) => {
                let frequency = query_frequency.then(|| -> rusqlite::Result<f64> {
                    self.connection.prepare_cached("SELECT frequency FROM tokens_view WHERE token_id = ?1")?
                        .query_row([id], |row| row.get::<_, f64>(0))
                }).transpose()?;

                Ok(Some(Token {
                    id,
                    value: token,
                    occurences,
                    frequency
                }))
            },

            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(err) => anyhow::bail!(err)
        }
    }

    /// Insert token to the database.
    ///
    /// Return id of inserted token.
    pub fn insert_token(&self, token: impl AsRef<str>) -> anyhow::Result<Token> {
        match self.query_token_by_value(token.as_ref(), false)? {
            Some(mut token) => {
                self.connection.prepare_cached("UPDATE tokens SET occurences = occurences + 1 WHERE id = ?1")?
                    .execute([token.id])?;

                token.occurences += 1;

                Ok(token)
            }

            None => {
                self.connection.prepare_cached("INSERT INTO tokens (value, occurences) VALUES (?1, 1)")?
                    .execute([token.as_ref()])?;

                Ok(Token {
                    id: self.connection.last_insert_rowid(),
                    value: token.as_ref().to_string(),
                    occurences: 1,
                    frequency: None
                })
            }
        }
    }

    /// Iterate over all tokens stored in the database.
    ///
    /// Return amount of read tokens.
    pub fn for_each(&self, query_frequency: bool, mut callback: impl FnMut(Token) -> anyhow::Result<()>) -> anyhow::Result<u64> {
        let mut tokens = 0;

        let mut query = if query_frequency {
            self.connection.prepare_cached("SELECT token_id, token_value, current_occurences, frequency FROM tokens_view ORDER BY id ASC")?
        } else {
            self.connection.prepare_cached("SELECT id, value, occurences, NULL as frequency FROM tokens ORDER BY id ASC")?
        };

        query.query_map([], move |row| {
            Ok(Token {
                id: row.get::<_, i64>(0)?,
                value: row.get::<_, String>(1)?,
                occurences: row.get::<_, i64>(2)?,
                frequency: row.get::<_, Option<f64>>(3)?
            })
        })?.try_for_each(|token| {
            tokens += 1;

            callback(token?)
        })?;

        Ok(tokens)
    }
}

#[test]
fn test_tokens_database() -> anyhow::Result<()> {
    let _ = std::fs::remove_file("tokens_database.db");

    let db = Database::open("tokens_database.db", 4096)?;

    assert!(db.query_token_by_value("hello", false)?.is_none());

    db.insert_token("hello")?;
    db.insert_token("world")?;

    assert_eq!(db.query_token_by_value("hello", false)?.unwrap().id, 1);
    assert_eq!(db.query_token_by_value("world", false)?.unwrap().id, 2);

    assert_eq!(db.query_token_by_value("hello", true)?.unwrap().frequency, Some(0.5));
    assert_eq!(db.query_token_by_value("world", true)?.unwrap().frequency, Some(0.5));

    let _ = std::fs::remove_file("tokens_database.db");

    Ok(())
}
