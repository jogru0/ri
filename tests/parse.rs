use ri::parse::{evaluate, to_tokens, IntConstant};

#[test]
fn punkt_vor_strich() {
    let s = "   1+2* 3   ;  ";

    let mut tokens = to_tokens(s).unwrap();

    let ast = tokens.parse_expr().unwrap();

    assert!(tokens.is_fully_parsed());

    let value = evaluate(ast).unwrap();

    assert_eq!(value, IntConstant::Small(7));
}

#[test]
fn punkt_vor_strich_ändert_nix() {
    let s = "1 *2 +3;";

    let mut tokens = to_tokens(s).unwrap();

    let ast = tokens.parse_expr().unwrap();

    assert!(tokens.is_fully_parsed());

    let value = evaluate(ast).unwrap();

    assert_eq!(value, IntConstant::Small(5));
}

#[test]
fn punkt_strich_punkt() {
    let s = " 0 * 10+10 * 0 ;";

    let mut tokens = to_tokens(s).unwrap();

    let ast = tokens.parse_expr().unwrap();

    assert!(tokens.is_fully_parsed());

    let value = evaluate(ast).unwrap();

    assert_eq!(value, IntConstant::Small(0));
}

#[test]
fn strich_punkt_strich() {
    let s = "1 + 10 *     10 

    
    + 4;";

    let mut tokens = to_tokens(s).unwrap();

    let ast = tokens.parse_expr().unwrap();

    assert!(tokens.is_fully_parsed());

    let value = evaluate(ast).unwrap();

    assert_eq!(value, IntConstant::Small(105));
}

#[test]
fn minus_minus() {
    let s = " 10 - 1 -\t\n 1;";

    let mut tokens = to_tokens(s).unwrap();

    let ast = tokens.parse_expr().unwrap();

    assert!(tokens.is_fully_parsed());

    let value = evaluate(ast).unwrap();

    assert_eq!(value, IntConstant::Small(8));
}
