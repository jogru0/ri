use std::{env::args, fmt::Debug, path::Path};

use log::error;
use ri::parse::{ModuleError, Modules, RuntimeError};
use thiserror::Error;

#[derive(Error)]
enum MainError {
    #[error("no source file provided")]
    NoSourceFile,
    #[error("module error: {0}")]
    ModuleError(#[from] ModuleError),
    #[error("Runtime Error: {0}")]
    RuntimeError(#[from] RuntimeError),
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

    let modules = Modules::from_entry_file(Path::new(path))?;

    let result = modules.evaluate_main()?;

    println!("Result: {result}");

    Ok(())
}
