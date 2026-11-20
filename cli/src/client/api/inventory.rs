use crate::client::error::Result;
use crate::client::http::HttpClient;
use crate::client::types::{Inventory, InventoryFilter, InventoryRegion};

#[derive(Debug, Clone)]
pub struct InventoryApi {
    http: HttpClient,
}

impl InventoryApi {
    pub(crate) fn new(http: HttpClient) -> Self {
        Self { http }
    }

    pub async fn list(&self, filter: &InventoryFilter) -> Result<Vec<Inventory>> {
        self.http.get_query("/inventory", &query(filter)).await
    }

    pub async fn list_for_org(
        &self,
        org: &str,
        filter: &InventoryFilter,
    ) -> Result<Vec<Inventory>> {
        self.http
            .get_query(&format!("/orgs/{org}/inventory"), &query(filter))
            .await
    }

    pub async fn regions(&self) -> Result<Vec<InventoryRegion>> {
        self.http.get("/region").await
    }
}

fn query(filter: &InventoryFilter) -> Vec<(&'static str, String)> {
    let mut query = Vec::new();
    if let Some(resource_type) = &filter.resource_type {
        query.push(("type", resource_type.clone()));
    }
    if let Some(gpu) = filter.gpu {
        query.push(("gpu", gpu.to_string()));
    }
    if let Some(region) = &filter.region {
        query.push(("region", region.clone()));
    }
    query
}
