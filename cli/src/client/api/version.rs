use crate::client::error::Result;
use crate::client::http::HttpClient;
use crate::client::types::Version;

#[derive(Debug, Clone)]
pub struct VersionApi {
    http: HttpClient,
}

impl VersionApi {
    pub(crate) fn new(http: HttpClient) -> Self {
        Self { http }
    }

    pub async fn get(&self) -> Result<Version> {
        self.http.get("/version").await
    }
}
