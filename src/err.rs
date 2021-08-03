use std::error;
use std::fmt;

#[derive(Debug)]
pub enum DeqError {
    Device,
    Operation,
    Format,
}

impl fmt::Display for DeqError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let msg = match *self {
            DeqError::Device => "invalid device id",
            DeqError::Operation => "invalid user operation",
            DeqError::Format => "invalid stream format",
        };
        f.write_str(msg)
    }
}

impl error::Error for DeqError {}
