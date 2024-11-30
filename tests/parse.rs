use ri::parse::{evaluate_debug, Constant, FunId, IntConstant, TokenStream, Tokens, Ty};

#[test]
fn punkt_vor_strich() {
    let s = "   1+2* 3     ";

    let tokens = Tokens::from_code(s).unwrap();

    let mut ts = TokenStream::new(&tokens);

    let expr = ts.parse_expr().unwrap();

    assert!(ts.is_fully_parsed());

    let value = evaluate_debug(expr, Ty::Int, ts.expressions).unwrap();

    assert_eq!(value, Constant::Int(IntConstant::Small(7)));
}

#[test]
fn punkt_vor_strich_ändert_nix() {
    let s = "1 *2 +3";

    let tokens = Tokens::from_code(s).unwrap();

    let mut ts = TokenStream::new(&tokens);

    let expr = ts.parse_expr().unwrap();

    assert!(ts.is_fully_parsed());

    let value = evaluate_debug(expr, Ty::Int, ts.expressions).unwrap();

    assert_eq!(value, Constant::Int(IntConstant::Small(5)));
}

#[test]
fn punkt_strich_punkt() {
    let s = " 0 * 10+10 * 0 ";

    let tokens = Tokens::from_code(s).unwrap();

    let mut ts = TokenStream::new(&tokens);

    let expr = ts.parse_expr().unwrap();

    assert!(ts.is_fully_parsed());

    let value = evaluate_debug(expr, Ty::Int, ts.expressions).unwrap();

    assert_eq!(value, Constant::Int(IntConstant::Small(0)));
}

#[test]
fn strich_punkt_strich() {
    let s = "1 + 10 *     10 

    
    + 4";

    let tokens = Tokens::from_code(s).unwrap();

    let mut ts = TokenStream::new(&tokens);

    let expr = ts.parse_expr().unwrap();

    assert!(ts.is_fully_parsed());

    let value = evaluate_debug(expr, Ty::Int, ts.expressions).unwrap();

    assert_eq!(value, Constant::Int(IntConstant::Small(105)));
}

#[test]
fn minus_minus() {
    let s = " 10 - 1 -\t\n 1";

    let tokens = Tokens::from_code(s).unwrap();

    let mut ts = TokenStream::new(&tokens);

    let expr = ts.parse_expr().unwrap();

    assert!(ts.is_fully_parsed());

    let value = evaluate_debug(expr, Ty::Int, ts.expressions).unwrap();

    assert_eq!(value, Constant::Int(IntConstant::Small(8)));
}

#[test]
fn fun_sum() {
    let s = "
    fun sum(a: int, b: int) -> int {
        return a + b;
    }
    
    
    ";

    let tokens = Tokens::from_code(s).unwrap();

    let ast = tokens.parse().unwrap();

    let value = ast
        .evaluate_fun(
            FunId(0),
            vec![
                Constant::Int(IntConstant::Small(4)),
                Constant::Int(IntConstant::Small(5)),
            ],
        )
        .unwrap();

    assert_eq!(value, Constant::Int(IntConstant::Small(9)));
}
