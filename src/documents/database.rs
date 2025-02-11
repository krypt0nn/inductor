use crate::prelude::*;

use std::path::Path;
use std::sync::{Arc, Mutex};

use rusqlite::Connection;

use burn::data::dataloader::Dataset;

#[derive(Debug, Clone)]
/// SQLite database for storing raw documents.
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
            CREATE TABLE IF NOT EXISTS documents (
                id      INTEGER NOT NULL,
                input   BLOB    NOT NULL,
                context BLOB    NOT NULL,
                output  BLOB    NOT NULL,

                PRIMARY KEY (id)
            );

            INSERT OR IGNORE INTO documents (id, input, context, output) VALUES (0, '', '', '');
        ")?;

        Ok(Self {
            connection: Arc::new(Mutex::new(connection))
        })
    }

    /// Get document from the database.
    ///
    /// Guaranteed to return `Ok(None)` if document with this id doesn't exist.
    pub fn get(&self, id: i64) -> anyhow::Result<Option<Document>> {
        let connection = self.connection.lock()
            .map_err(|_| anyhow::anyhow!("Failed to lock sqlite connection"))?;

        let row = connection.prepare_cached("SELECT input, context, output FROM documents WHERE id = ?1")?
            .query_row([id], |row| {
                let input   = row.get::<_, Vec<u8>>(0)?;
                let context = row.get::<_, Vec<u8>>(1)?;
                let output  = row.get::<_, Vec<u8>>(2)?;

                Ok((input, context, output))
            });

        let row = match row {
            Ok(row) => row,
            Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(None),
            Err(err) => anyhow::bail!(err)
        };

        let (input, context, output) = row;

        let input   = lz4_flex::decompress_size_prepended(&input)?;
        let context = lz4_flex::decompress_size_prepended(&context)?;
        let output  = lz4_flex::decompress_size_prepended(&output)?;

        Ok(Some(Document {
            input: String::from_utf8(input)?,
            context: String::from_utf8(context)?,
            output: String::from_utf8(output)?
        }))
    }

    /// Insert document to the database.
    pub fn insert(&self, document: &Document) -> anyhow::Result<i64> {
        let connection = self.connection.lock()
            .map_err(|_| anyhow::anyhow!("Failed to lock sqlite connection"))?;

        connection.prepare_cached("INSERT INTO documents (input, context, output) VALUES (?1, ?2, ?3)")?
            .execute([
                lz4_flex::compress_prepend_size(document.input.as_bytes()),
                lz4_flex::compress_prepend_size(document.context.as_bytes()),
                lz4_flex::compress_prepend_size(document.output.as_bytes())
            ])?;

        Ok(connection.last_insert_rowid())
    }

    #[inline]
    /// Get amount of documents stored in the database.
    pub fn len(&self) -> anyhow::Result<u64> {
        let connection = self.connection.lock()
            .map_err(|_| anyhow::anyhow!("Failed to lock sqlite connection"))?;

        let len = connection.prepare("SELECT COUNT(id) FROM documents")?
            .query_row([], |row| row.get::<_, u64>(0))?;

        Ok(len + 1)
    }

    #[inline]
    pub fn is_empty(&self) -> anyhow::Result<bool> {
        Ok(self.len()? < 2)
    }

    /// Iterate over all documents stored in the database.
    ///
    /// Return amount of read tokens.
    pub fn for_each(&self, mut callback: impl FnMut(Document) -> anyhow::Result<()>) -> anyhow::Result<u64> {
        let mut tokens = 0;

        while let Some(document) = self.get(tokens + 1)? {
            callback(document)?;

            tokens += 1;
        }

        Ok(tokens as u64)
    }
}

impl Dataset<Document> for Database {
    fn get(&self, index: usize) -> Option<Document> {
        self.get(index as i64).unwrap_or_default()
    }

    fn len(&self) -> usize {
        self.len().unwrap_or_default() as usize
    }

    fn is_empty(&self) -> bool {
        self.is_empty().unwrap_or(true)
    }
}

#[test]
fn test_documents_database() -> anyhow::Result<()> {
    let _ = std::fs::remove_file("documents_database.db");

    let db = Database::open("documents_database.db", 4096)?;

    db.insert(&Document::new("Test document 1"))?;
    db.insert(&Document::new("Test document 2"))?;
    db.insert(&Document::new("Test document 3"))?;

    assert_eq!(db.get(1)?.unwrap().output, "Test document 1");
    assert_eq!(db.get(2)?.unwrap().output, "Test document 2");
    assert_eq!(db.get(3)?.unwrap().output, "Test document 3");

    let mut i = 1;

    let n = db.for_each(move |document| {
        assert_eq!(document.output, format!("Test document {i}"));

        i += 1;

        Ok(())
    })?;

    assert_eq!(n, 3);

    let _ = std::fs::remove_file("documents_database.db");

    Ok(())
}
