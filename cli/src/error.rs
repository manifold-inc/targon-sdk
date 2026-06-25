use std::io;

use crate::client::ClientError;

pub type Result<T> = std::result::Result<T, CliError>;

#[derive(Debug, thiserror::Error)]
pub enum CliError {
    #[error(transparent)]
    Client(#[from] ClientError),

    #[error("not authenticated; run `targon auth login` or set TARGON_API_KEY")]
    NotAuthenticated,

    #[error("{0} is required; pass it as a flag when running non-interactively")]
    NotInteractive(String),

    #[error("{0}")]
    Config(String),

    #[error("cancelled")]
    Cancelled,

    #[error(transparent)]
    Io(#[from] io::Error),
}

impl CliError {
    pub fn exit_code(&self) -> u8 {
        match self {
            CliError::NotAuthenticated => 3,
            CliError::Client(ClientError::Api { status, .. }) => match status {
                401 | 403 => 3,
                404 => 4,
                _ => 1,
            },
            CliError::Config(_) | CliError::NotInteractive(_) => 2,
            CliError::Cancelled => 130,
            _ => 1,
        }
    }
}
