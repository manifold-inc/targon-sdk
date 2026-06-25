use std::result::Result as StdResult;

pub type Result<T> = StdResult<T, ClientError>;

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("request failed: {0}")]
    Transport(#[from] reqwest::Error),

    #[error("api error {status} ({reason}): {message}")]
    Api {
        status: u16,
        reason: String,
        message: String,
    },

    #[error("failed to decode response: {0}")]
    Decode(String),

    #[error("invalid client config: {0}")]
    InvalidConfig(String),
}
