//! Crate-wide error type.

use thiserror::Error;

/// All errors the simulator can raise.
#[derive(Debug, Error)]
pub enum SplatErrors {
    #[error("Invalid parameter '{name}': {reason}")]
    InvalidParam {
        /// Parameter name, in splatter's spelling
        name: String,
        /// What is wrong with it
        reason: String,
    },

    #[error("'{name}' = '{value}' is not supported by this port.")]
    Unsupported {
        /// Parameter name, in splatter's spelling
        name: &'static str,
        /// The rejected value
        value: String,
    },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("HDF5 error: {0}")]
    Hdf5(#[from] hdf5::Error),
}

/// Shorthand for an [`SplatErrors::InvalidParam`].
///
/// ### Params
///
/// * `name` - Parameter name
/// * `reason` - What is wrong with it
///
/// ### Returns
///
/// The error value.
pub(crate) fn invalid(name: impl Into<String>, reason: impl Into<String>) -> SplatErrors {
    SplatErrors::InvalidParam {
        name: name.into(),
        reason: reason.into(),
    }
}
