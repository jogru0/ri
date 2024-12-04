use indexmap::IndexSet;
use ri::parse::{evaluate_debug, FunId, TokenStream, Tokens, Ty, VariableValues};

#[test]
fn punkt_vor_strich() {
    let s = "   1+2* 3     ";

    let tokens = Tokens::from_code(s, "mock.ri".into()).unwrap();

    let fun_names = IndexSet::new();
    let mut ts = TokenStream::new(&tokens, &fun_names);

    let expr = ts.parse_expr().unwrap();

    assert!(ts.is_fully_parsed());

    let value = evaluate_debug(expr, Ty::Int, ts.expressions).unwrap();

    assert_eq!(value, 7.into());
}

#[test]
fn punkt_vor_strich_ändert_nix() {
    let s = "1 *2 +3";

    let tokens = Tokens::from_code(s, "mock.ri".into()).unwrap();

    let fun_names = IndexSet::new();
    let mut ts = TokenStream::new(&tokens, &fun_names);

    let expr = ts.parse_expr().unwrap();

    assert!(ts.is_fully_parsed());

    let value = evaluate_debug(expr, Ty::Int, ts.expressions).unwrap();

    assert_eq!(value, 5.into());
}

#[test]
fn punkt_strich_punkt() {
    let s = " 0 * 10+10 * 0 ";

    let tokens = Tokens::from_code(s, "mock.ri".into()).unwrap();

    let fun_names = IndexSet::new();
    let mut ts = TokenStream::new(&tokens, &fun_names);

    let expr = ts.parse_expr().unwrap();

    assert!(ts.is_fully_parsed());

    let value = evaluate_debug(expr, Ty::Int, ts.expressions).unwrap();

    assert_eq!(value, 0.into());
}

#[test]
fn strich_punkt_strich() {
    let s = "1 + 10 *     10

    + 4";

    let tokens = Tokens::from_code(s, "mock.ri".into()).unwrap();

    let fun_names = IndexSet::new();
    let mut ts = TokenStream::new(&tokens, &fun_names);

    let expr = ts.parse_expr().unwrap();

    assert!(ts.is_fully_parsed());

    let value = evaluate_debug(expr, Ty::Int, ts.expressions).unwrap();

    assert_eq!(value, 105.into());
}

#[test]
fn minus_minus() {
    let s = " 10 - 1 -\t\n 1";

    let tokens = Tokens::from_code(s, "mock.ri".into()).unwrap();

    let fun_names = IndexSet::new();
    let mut ts = TokenStream::new(&tokens, &fun_names);

    let expr = ts.parse_expr().unwrap();

    assert!(ts.is_fully_parsed());

    let value = evaluate_debug(expr, Ty::Int, ts.expressions).unwrap();

    assert_eq!(value, 8.into());
}

#[test]
fn fun_sum() {
    let s = "
    fun sum(a: int, b: int) -> int {
        return a + b;
    }

    ";

    let tokens = Tokens::from_code(s, "mock.ri".into()).unwrap();

    let ast = tokens.parse().unwrap();

    let value = ast
        .evaluate_fun(
            FunId(0),
            vec![4.into(), 5.into()],
            &mut VariableValues::new(),
        )
        .unwrap();

    assert_eq!(value, 9.into());
}
