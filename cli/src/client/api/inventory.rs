use crate::client::error::Result;
use crate::client::http::HttpClient;
use crate::client::types::{Inventory, InventoryFilter};

#[derive(Debug, Clone)]
pub struct InventoryApi {
    http: HttpClient,
}

impl InventoryApi {
    pub(crate) fn new(http: HttpClient) -> Self {
        Self { http }
    }

    pub async fn list(&self, filter: &InventoryFilter) -> Result<Vec<Inventory>> {
        let mut query = Vec::new();
        if let Some(resource_type) = &filter.resource_type {
            query.push(("type", resource_type.clone()));
        }
        if let Some(gpu) = filter.gpu {
            query.push(("gpu", gpu.to_string()));
        }
        self.http.get_query("/inventory", &query).await
    }
}
