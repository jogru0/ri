use std::error::Error;

use ri::parse::{
    evaluate_debug, Constant, Expressions, ModuleHeaders, ModuleId, TokenStream, Tokens, Ty, Ty2,
};

fn eval(expr: &str) -> Result<Constant, Box<dyn Error>> {
    let tokens = Tokens::from_code(expr, "mock.ri".into())?;

    let mut expressions = Expressions::new();
    let module_headers = ModuleHeaders(Vec::new());
    let mut ts = TokenStream::new(
        &tokens,
        &module_headers,
        ModuleId(0),
        &mut expressions,
        // &mod_names,
    );

    let expr = ts.parse_expr()?;

    assert!(ts.is_fully_parsed());

    Ok(evaluate_debug(expr, Ty::Int, Ty2::Int, expressions)?)
}

#[test]
fn punkt_vor_strich() {
    let expr_str = "   1+2* 3     ";
    let expected = 7;
    assert_eq!(eval(expr_str).unwrap(), expected.into());
}

#[test]
fn punkt_vor_strich_ändert_nix() {
    let expr_str = "1 *2 +3";
    let expected = 5;
    assert_eq!(eval(expr_str).unwrap(), expected.into());
}

#[test]
fn punkt_strich_punkt() {
    let expr_str = " 0 * 10+10 * 0 ";
    let expected = 0;
    assert_eq!(eval(expr_str).unwrap(), expected.into());
}

#[test]
fn strich_punkt_strich() {
    let expr_str = "1 + 10 *     10

    + 4";
    let expected = 105;
    assert_eq!(eval(expr_str).unwrap(), expected.into());
}

#[test]
fn minus_minus() {
    let expr_str = " 10 - 1 -\t\n 1";
    let expected = 8;
    assert_eq!(eval(expr_str).unwrap(), expected.into());
}

//TODO unit test to parse and eval a fun
