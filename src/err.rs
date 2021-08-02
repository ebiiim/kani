use std::error;
use std::fmt;

#[derive(PartialEq, Eq, Debug)]
pub enum DeqError {
    InvalidDevice,
    InvalidOperation,
}

impl fmt::Display for DeqError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let msg = match *self {
            DeqError::InvalidDevice => "invalid device id",
            DeqError::InvalidOperation => "invalid user operation",
        };
        f.write_str(msg)
    }
}

impl error::Error for DeqError {}
