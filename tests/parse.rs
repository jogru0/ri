use indexmap::indexmap;
use ri::parse::{evaluate, evaluate_fun, to_tokens, IntConstant, VariableValues, Word};

#[test]
fn punkt_vor_strich() {
    let s = "   1+2* 3     ";

    let mut tokens = to_tokens(s).unwrap();

    let ast = tokens.parse_expr().unwrap();

    assert!(tokens.is_fully_parsed());

    let value = evaluate(ast).unwrap();

    assert_eq!(value, IntConstant::Small(7));
}

#[test]
fn punkt_vor_strich_ändert_nix() {
    let s = "1 *2 +3";

    let mut tokens = to_tokens(s).unwrap();

    let ast = tokens.parse_expr().unwrap();

    assert!(tokens.is_fully_parsed());

    let value = evaluate(ast).unwrap();

    assert_eq!(value, IntConstant::Small(5));
}

#[test]
fn punkt_strich_punkt() {
    let s = " 0 * 10+10 * 0 ";

    let mut tokens = to_tokens(s).unwrap();

    let ast = tokens.parse_expr().unwrap();

    assert!(tokens.is_fully_parsed());

    let value = evaluate(ast).unwrap();

    assert_eq!(value, IntConstant::Small(0));
}

#[test]
fn strich_punkt_strich() {
    let s = "1 + 10 *     10 

    
    + 4";

    let mut tokens = to_tokens(s).unwrap();

    let ast = tokens.parse_expr().unwrap();

    assert!(tokens.is_fully_parsed());

    let value = evaluate(ast).unwrap();

    assert_eq!(value, IntConstant::Small(105));
}

#[test]
fn minus_minus() {
    let s = " 10 - 1 -\t\n 1";

    let mut tokens = to_tokens(s).unwrap();

    let ast = tokens.parse_expr().unwrap();

    assert!(tokens.is_fully_parsed());

    let value = evaluate(ast).unwrap();

    assert_eq!(value, IntConstant::Small(8));
}

#[test]
fn fun_sum() {
    let s = "
    fun sum(a: int, b: int) -> int {
        return a + b;
    }
    
    
    ";

    let mut tokens = to_tokens(s).unwrap();

    let fun = tokens.parse_fun().unwrap();

    assert!(tokens.is_fully_parsed());

    let value = evaluate_fun(
        fun,
        VariableValues::new(
            indexmap! {Word::new("a".into()) => IntConstant::Small(4), Word::new("b".into()) => IntConstant::Small(5)},
        ),
    )
    .unwrap();

    assert_eq!(value, IntConstant::Small(9));
}
