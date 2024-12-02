use std::{env::args, fmt::Debug, fs::read_to_string, io::Error};

use log::error;
use ri::parse::{ModuleError, ParseError, RuntimeError, SourceTokenizeError, TokenStream, Tokens};
use thiserror::Error;

#[derive(Error)]
enum MainError {
    #[error("no source file provided")]
    NoSourceFile,
    #[error("Tokenize Error: {0}")]
    TokenizeError(#[from] SourceTokenizeError),
    #[error("Parse Error: {0}")]
    ParseError(#[from] ParseError),
    #[error("Runtime Error: {0}")]
    RuntimeError(#[from] RuntimeError),
    #[error("Module Error: {0}")]
    ModuleError(#[from] ModuleError),
}

// Make main use Display, not Debug, for error reporting.
impl Debug for MainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s: String = format!("{self}")
            .chars()
            .map(|c| if c == '\'' { '`' } else { c })
            .collect();
        write!(f, "{}", s)
    }
}

fn main() -> Result<(), MainError> {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        // .filter_module("ri", log::LevelFilter::Debug)
        .init();

    let args: Vec<String> = args().collect();

    let path = args.get(1).ok_or(MainError::NoSourceFile)?;

    let tokens = TokenStream::new(path.clone())?;

    let ast = tokens.parse()?;

    let result = ast.evaluate_main()?;

    println!("Result: {result}");

    Ok(())
}
