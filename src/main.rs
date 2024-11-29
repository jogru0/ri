use std::{env::args, fmt::Debug, fs::read_to_string, io::Error};

use ri::parse::{tokenize, ParseError, RuntimeError, TokenizeError};
use thiserror::Error;

#[derive(Error)]
enum MainError {
    #[error("no source file provided")]
    NoSourceFile,
    #[error("Tokenize Error: {0}")]
    TokenizeError(#[from] TokenizeError),
    #[error("Parse Error: {0}")]
    ParseError(#[from] ParseError),
    #[error("Runtime Error: {0}")]
    RuntimeError(#[from] RuntimeError),
    #[error("Io Error for '{0}': {1}")]
    IoError(String, Error),
}

// Make main use Display, not Debug, for error reporting.
impl Debug for MainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

fn main() -> Result<(), MainError> {
    let args: Vec<String> = args().collect();

    let path = args.get(1).ok_or(MainError::NoSourceFile)?;

    let code = read_to_string(&args[1]).map_err(|err| MainError::IoError(path.into(), err))?;

    let tokens = tokenize(&code)?;

    let ast = tokens.parse()?;

    let result = ast.evaluate_main()?;

    println!("Result: {result}");

    Ok(())
}
