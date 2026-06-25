use bytes::Bytes;
use futures_util::Stream;
use reqwest::header::{HeaderMap, HeaderValue, ACCEPT, AUTHORIZATION};
use reqwest::{Client, Method, RequestBuilder, Response};
use serde::de::DeserializeOwned;
use serde::Serialize;

use super::config::ClientConfig;
use super::error::{ClientError, Result};

#[derive(Debug, Clone)]
pub struct HttpClient {
    inner: Client,
    config: ClientConfig,
}

impl HttpClient {
    pub fn new(config: ClientConfig) -> Result<Self> {
        let mut headers = HeaderMap::new();
        let mut auth = HeaderValue::from_str(&format!("Bearer {}", config.api_key))
            .map_err(|e| ClientError::InvalidConfig(e.to_string()))?;
        auth.set_sensitive(true);
        headers.insert(AUTHORIZATION, auth);
        headers.insert(ACCEPT, HeaderValue::from_static("application/json"));

        let inner = Client::builder()
            .default_headers(headers)
            .user_agent(concat!("targon-cli/", env!("CARGO_PKG_VERSION")))
            .build()?;

        Ok(Self { inner, config })
    }

    pub async fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        self.send_json(self.request(Method::GET, path)).await
    }

    pub async fn get_query<T: DeserializeOwned>(
        &self,
        path: &str,
        query: &[(&str, String)],
    ) -> Result<T> {
        self.send_json(self.request(Method::GET, path).query(query)).await
    }

    pub async fn post<T: DeserializeOwned, B: Serialize>(&self, path: &str, body: &B) -> Result<T> {
        self.send_json(self.request(Method::POST, path).json(body)).await
    }

    pub async fn post_empty<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        self.send_json(self.request(Method::POST, path)).await
    }

    pub async fn patch<T: DeserializeOwned, B: Serialize>(&self, path: &str, body: &B) -> Result<T> {
        self.send_json(self.request(Method::PATCH, path).json(body)).await
    }

    pub async fn put<T: DeserializeOwned, B: Serialize>(&self, path: &str, body: &B) -> Result<T> {
        self.send_json(self.request(Method::PUT, path).json(body)).await
    }

    pub async fn put_empty<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        self.send_json(self.request(Method::PUT, path)).await
    }

    pub async fn delete(&self, path: &str) -> Result<()> {
        let resp = self
            .request(Method::DELETE, path)
            .timeout(self.config.timeout)
            .send()
            .await?;
        self.check(resp).await?;
        Ok(())
    }

    pub async fn get_text(&self, path: &str, query: &[(&str, String)]) -> Result<String> {
        let resp = self
            .request(Method::GET, path)
            .query(query)
            .timeout(self.config.timeout)
            .send()
            .await?;
        let resp = self.check(resp).await?;
        Ok(resp.text().await?)
    }

    pub async fn stream(
        &self,
        path: &str,
        query: &[(&str, String)],
    ) -> Result<impl Stream<Item = reqwest::Result<Bytes>>> {
        // Log following stays open indefinitely, so no per-request timeout is applied.
        let resp = self
            .request(Method::GET, path)
            .query(query)
            .header(ACCEPT, "text/plain")
            .send()
            .await?;
        let resp = self.check(resp).await?;
        Ok(resp.bytes_stream())
    }

    fn request(&self, method: Method, path: &str) -> RequestBuilder {
        self.inner.request(method, self.config.url(path))
    }

    async fn send_json<T: DeserializeOwned>(&self, req: RequestBuilder) -> Result<T> {
        let resp = req.timeout(self.config.timeout).send().await?;
        let resp = self.check(resp).await?;
        let bytes = resp.bytes().await?;
        serde_json::from_slice(&bytes).map_err(|e| ClientError::Decode(e.to_string()))
    }

    async fn check(&self, resp: Response) -> Result<Response> {
        if resp.status().is_success() {
            return Ok(resp);
        }
        Err(error_from_response(resp).await)
    }
}

#[derive(serde::Deserialize)]
struct ErrorBody {
    #[serde(default)]
    error: String,
    #[serde(default)]
    reason: String,
}

async fn error_from_response(resp: Response) -> ClientError {
    let status = resp.status().as_u16();
    let body = resp.text().await.unwrap_or_default();

    match serde_json::from_str::<ErrorBody>(&body) {
        Ok(parsed) if !parsed.error.is_empty() => ClientError::Api {
            status,
            reason: parsed.reason,
            message: parsed.error,
        },
        _ => ClientError::Api {
            status,
            reason: String::new(),
            message: body,
        },
    }
}
