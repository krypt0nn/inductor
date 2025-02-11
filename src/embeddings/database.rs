use std::path::Path;
use std::sync::{Arc, Mutex};

use rusqlite::Connection;

use super::prelude::*;

#[inline]
/// Parse vector of floats from the bytes slice.
fn parse_embedding(bytes: &[u8]) -> anyhow::Result<Vec<f32>> {
    let n = bytes.len();

    if n % 4 != 0 {
        anyhow::bail!("Trying to query token embedding with different float type");
    }

    let mut embedding = Vec::with_capacity(n / 4);
    let mut be_bytes = [0; 4];

    let mut k = 0;

    while k < n {
        be_bytes.copy_from_slice(&bytes[k..k + 4]);

        embedding.push(f32::from_be_bytes(be_bytes));

        k += 4;
    }

    Ok(embedding)
}

#[derive(Debug, Clone)]
/// SQLite database for storing tokens and their embeddings.
pub struct Database {
    connection: Arc<Mutex<Connection>>
}

impl Database {
    /// Open database with given cache size.
    /// Negative number means sqlite pages (1024 bytes), positive - bytes.
    pub fn open(path: impl AsRef<Path>, cache_size: i64) -> rusqlite::Result<Self> {
        let connection = Connection::open(path)?;

        connection.execute(&format!("PRAGMA cache_size = {cache_size};"), ())?;

        connection.execute_batch("
            CREATE TABLE IF NOT EXISTS tokens (
                id    INTEGER NOT NULL,
                value TEXT UNIQUE NOT NULL,

                PRIMARY KEY (id)
            );

            CREATE INDEX IF NOT EXISTS idx_tokens_value on tokens (value);

            CREATE TABLE IF NOT EXISTS embeddings (
                token_id  INTEGER NOT NULL,
                embedding BLOB NOT NULL,

                PRIMARY KEY (token_id),
                FOREIGN KEY (token_id) REFERENCES tokens (id)
            );

            INSERT OR IGNORE INTO tokens (id, value) VALUES (0, '');
            INSERT OR IGNORE INTO embeddings (token_id, embedding) VALUES (0, '');
        ")?;

        Ok(Self {
            connection: Arc::new(Mutex::new(connection))
        })
    }

    /// Insert token embedding to the database.
    pub fn insert_embedding(&self, token: impl AsRef<str>, embedding: &[f32]) -> anyhow::Result<()> {
        let mut embedding_bytes = vec![0; embedding.len() * 4];

        for (i, float) in embedding.iter().enumerate() {
            embedding_bytes[i * 4..(i + 1) * 4].copy_from_slice(&float.to_be_bytes());
        }

        let connection = self.connection.lock()
            .map_err(|_| anyhow::anyhow!("Failed to lock sqlite connection"))?;

        connection.prepare_cached("INSERT OR IGNORE INTO tokens (value) VALUES (?1)")?
            .execute([token.as_ref()])?;

        let token_id = connection.prepare_cached("SELECT id FROM tokens WHERE value = ?1")?
            .query_row([token.as_ref()], |row| row.get::<_, i64>(0))?;

        connection.prepare_cached("INSERT OR REPLACE INTO embeddings (token_id, embedding) VALUES (?1, ?2)")?
            .execute((token_id, embedding_bytes))?;

        Ok(())
    }

    /// Query token embedding from the database.
    ///
    /// Guaranteed to return `Ok(None)` if token is not stored.
    pub fn query_embedding(&self, token: impl AsRef<str>) -> anyhow::Result<Option<Vec<f32>>> {
        let connection = self.connection.lock()
            .map_err(|_| anyhow::anyhow!("Failed to lock sqlite connection"))?;

        let mut query = connection.prepare_cached("
            SELECT embedding FROM embeddings
            INNER JOIN tokens
            ON embeddings.token_id = tokens.id
            WHERE tokens.value = ?1
        ")?;

        let embedding = query.query_row([token.as_ref()], |row| row.get::<_, Vec<u8>>(0));

        let embedding = match embedding {
            Ok(embedding_bytes) => parse_embedding(&embedding_bytes)?,
            Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(None),
            Err(err) => anyhow::bail!(err)
        };

        Ok(Some(embedding))
    }

    /// Find token in the database with the closest to given embedding.
    ///
    /// Guaranteed to return some value unless the database is empty.
    pub fn find_token(&self, embedding: &[f32]) -> anyhow::Result<Option<String>> {
        let connection = self.connection.lock()
            .map_err(|_| anyhow::anyhow!("Failed to lock sqlite connection"))?;

        let mut rows = connection.prepare_cached("SELECT value FROM tokens")?
            .query_map((), |row| row.get::<_, String>(0))?
            .map(|token| -> anyhow::Result<_> {
                let token = token?;
                let embedding = self.query_embedding(&token)?;

                Ok((token, embedding))
            })
            .flat_map(|row| {
                match row {
                    Ok((token, Some(embedding))) => Some(Ok((token, embedding))),
                    Ok((_, None)) => None,
                    Err(err) => Some(Err(err))
                }
            })
            .map(|row| -> anyhow::Result<_> {
                let (token, token_embedding) = row?;

                Ok((token, cosine_similarity::<EMBEDDING_SIZE>(&token_embedding, embedding)))
            })
            .collect::<Result<Vec<_>, _>>()?;

        drop(connection);

        let Some(mut closest_token) = rows.pop() else {
            return Ok(None);
        };

        for (token, similarity) in rows {
            if similarity >= 1.0 {
                return Ok(Some(token));
            }

            if similarity > closest_token.1 {
                closest_token = (token, similarity);
            }
        }

        Ok(Some(closest_token.0))
    }

    /// Iterate over all tokens stored in the database.
    ///
    /// Return amount of read tokens.
    pub fn for_each(&self, mut callback: impl FnMut(String, Vec<f32>) -> anyhow::Result<()>) -> anyhow::Result<u64> {
        let mut tokens = 0;

        let connection = self.connection.lock()
            .map_err(|_| anyhow::anyhow!("Failed to lock sqlite connection"))?;

        let mut query = connection.prepare_cached("
            SELECT tokens.value, embeddings.embedding
            FROM tokens
            INNER JOIN embeddings
            ON tokens.id = embeddings.token_id
            ORDER BY tokens.id ASC
        ")?;

        query.query_map([], |row| {
            let token = row.get::<_, String>(0)?;
            let embedding = row.get::<_, Vec<u8>>(1)?;

            Ok((token, embedding))
        })?.try_for_each(|row| {
            let (token, embedding) = row?;

            tokens += 1;

            callback(token, parse_embedding(&embedding)?)
        })?;

        Ok(tokens)
    }
}

#[test]
fn test_tokens_database() -> anyhow::Result<()> {
    let _ = std::fs::remove_file("embeddings_database.db");

    let db = Database::open("embeddings_database.db", 4096)?;

    db.insert_embedding("hello", &[1.0, 2.0, 3.0])?;
    db.insert_embedding("world", &[1.0, 2.0, 4.0])?;

    assert_eq!(db.query_embedding("hello")?.as_deref(), Some([1.0, 2.0, 3.0].as_slice()));
    assert_eq!(db.query_embedding("world")?.as_deref(), Some([1.0, 2.0, 4.0].as_slice()));

    assert_eq!(db.find_token(&[1.0, 2.0, 2.0])?.as_deref(), Some("hello"));
    assert_eq!(db.find_token(&[1.0, 2.0, 5.0])?.as_deref(), Some("world"));

    let _ = std::fs::remove_file("embeddings_database.db");

    Ok(())
}
