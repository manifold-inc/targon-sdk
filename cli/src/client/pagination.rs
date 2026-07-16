use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default)]
pub struct Page {
    pub limit: Option<u32>,
    pub cursor: Option<String>,
}

impl Page {
    pub fn query(&self) -> Vec<(&'static str, String)> {
        let mut query = Vec::new();
        if let Some(limit) = self.limit {
            query.push(("limit", limit.to_string()));
        }
        if let Some(cursor) = &self.cursor {
            query.push(("cursor", cursor.clone()));
        }
        query
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct List<T> {
    #[serde(default = "Vec::new")]
    pub items: Vec<T>,
    #[serde(default)]
    pub next_cursor: Option<String>,
}
