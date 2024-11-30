use std::{
    cmp::Ordering,
    error::Error,
    fmt::Display,
    ops::{Add, Mul, Sub},
    sync::LazyLock,
};

use indexmap::{indexmap, map::Entry, IndexMap, IndexSet};
use thiserror::Error;
use word::Word;

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("not (yet) implemented: {0}")]
    NotImplememted(String),
    #[error("unknown variable '{0}'")]
    UnknownVariable(Word),
    //TODO: Nicer
    #[error("arguments '{0:?}' incompatible with parameters '{1:?}'")]
    IncompatibleParameters(Vec<Constant>, Variables),
    #[error("missing '{}'", Token::Return)]
    MisingReturn,
    #[error("unknown function '{0}'")]
    UnknownFunction(Word),
    #[error("type error: exoected '{0}', found '{1}'")]
    TypeError(Ty, Ty),
    #[error("invalid operation:'{0} {1} {2}'")]
    InvalidBinaryOperation(Constant, BinaryOperator, Constant),
    #[error("function'{0}' has return type '{1}', but evaluated to '{2}'")]
    WrongReturnType(Word, Ty, Ty),
    #[error("missing entry point (function '{}')", *MAIN)]
    NoEntryPoint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum IntConstant {
    Small(i128),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Constant {
    Int(IntConstant),
    Bool(bool),
    None,
}
impl Constant {
    fn get_bool(self) -> Result<bool, RuntimeError> {
        if let Constant::Bool(b) = self {
            Ok(b)
        } else {
            let actual = self.ty();
            let expected = Ty::Bool;
            assert_ne!(actual, expected);
            Err(RuntimeError::TypeError(expected, actual))
        }
    }

    fn ty(&self) -> Ty {
        match self {
            Constant::Int(_) => Ty::Int,
            Constant::Bool(_) => Ty::Bool,
            Constant::None => Ty::None,
        }
    }
}

impl Display for IntConstant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IntConstant::Small(i) => write!(f, "{i}"),
        }
    }
}

impl Display for Constant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Constant::Int(int) => write!(f, "{int}"),
            Constant::Bool(b) => write!(f, "{b}"),
            Constant::None => write!(f, "()"),
        }
    }
}

impl Sub for IntConstant {
    type Output = Result<IntConstant, RuntimeError>;

    fn sub(self, other: Self) -> Self::Output {
        match (self, other) {
            (IntConstant::Small(lhs), IntConstant::Small(rhs)) => lhs
                .checked_sub(rhs)
                .ok_or(RuntimeError::NotImplememted("numbers beyond i128".into()))
                .map(Self::Small),
        }
    }
}

impl Mul for IntConstant {
    type Output = Result<IntConstant, RuntimeError>;

    fn mul(self, other: Self) -> Self::Output {
        match (self, other) {
            (IntConstant::Small(lhs), IntConstant::Small(rhs)) => lhs
                .checked_mul(rhs)
                .ok_or(RuntimeError::NotImplememted("numbers beyond i128".into()))
                .map(Self::Small),
        }
    }
}

impl Add for IntConstant {
    type Output = Result<IntConstant, RuntimeError>;

    fn add(self, other: Self) -> Self::Output {
        match (self, other) {
            (IntConstant::Small(lhs), IntConstant::Small(rhs)) => lhs
                .checked_add(rhs)
                .ok_or(RuntimeError::NotImplememted("numbers beyond i128".into()))
                .map(Self::Small),
        }
    }
}

#[derive(Clone, Copy)]
pub enum Expr {
    Return(ExprId),
    Constant(Constant),
    Variable(VariableId),
    BinaryOp(ExprId, BinaryOperator, ExprId),
    Block(Block),
    //TODO: Test for block here
    If(ExprId, Block),
    Call(Call),
}

#[derive(Clone, Copy)]
pub struct VariableId(usize);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExprId(usize);

#[derive(Clone, Copy)]
//TODO pub field
pub struct FunId(pub usize);

#[derive(Clone, Copy)]
pub struct ExprRange {
    start: ExprId,
    end: ExprId,
}

#[derive(Clone, Copy)]
pub struct Block {
    statements: ExprRange,
}

#[derive(Clone, Copy)]
pub struct Call {
    fun_id: FunId,
    arguments: ExprRange,
}

impl Block {
    fn new(statements: ExprRange) -> Self {
        Self { statements }
    }
}

#[derive(Debug, Clone)]
pub struct Variables {
    set: IndexMap<Word, Variable>,
}
impl Variables {
    fn new() -> Self {
        Self {
            set: IndexMap::new(),
        }
    }

    fn insert_new(&mut self, variable: Variable) -> Result<(), ParseError> {
        match self.set.entry(variable.name.clone()) {
            indexmap::map::Entry::Occupied(_) => {
                Err(ParseError::AlreadyDefinedVariable(variable.name))
            }
            indexmap::map::Entry::Vacant(vacant_entry) => {
                vacant_entry.insert(variable);
                Ok(())
            }
        }
    }

    fn len(&self) -> usize {
        self.set.len()
    }
}

pub struct VariableValues {
    values: Vec<Constant>,
}

impl VariableValues {
    fn get(&self, variable_id: &VariableId) -> Result<Constant, RuntimeError> {
        self.values
            .get(variable_id.0)
            .ok_or_else(|| //RuntimeError::UnknownVariable(word.clone())
                panic!("how"))
            .cloned()
    }

    pub fn new(values: Vec<Constant>) -> Self {
        Self { values }
    }
}

#[must_use]
struct Evaluation {
    value: Constant,
    returning: bool,
}

impl Evaluation {
    fn returning(self) -> Self {
        Self {
            value: self.value,
            returning: true,
        }
    }

    fn unwrap_on_outer_layer(&self) -> Constant {
        self.value
    }

    fn some_or_please_return(&self) -> Option<Constant> {
        if self.returning {
            None
        } else {
            Some(self.value)
        }
    }
}

impl From<Constant> for Evaluation {
    fn from(val: Constant) -> Self {
        Evaluation {
            value: val,
            returning: false,
        }
    }
}

impl Add for Constant {
    type Output = Result<Constant, RuntimeError>;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Constant::Int(lhs), Constant::Int(rhs)) => Ok(Constant::Int((lhs + rhs)?)),
            (lhs, rhs) => Err(RuntimeError::InvalidBinaryOperation(
                lhs,
                BinaryOperator::Plus,
                rhs,
            )),
        }
    }
}

impl Mul for Constant {
    type Output = Result<Constant, RuntimeError>;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Constant::Int(lhs), Constant::Int(rhs)) => Ok(Constant::Int((lhs * rhs)?)),
            (lhs, rhs) => Err(RuntimeError::InvalidBinaryOperation(
                lhs,
                BinaryOperator::Times,
                rhs,
            )),
        }
    }
}

// impl Ord for IntConstant {
//     fn cmp(&self, other: &Self) -> Ordering {
//         match (self, other) {
//             (IntConstant::Small(lhs), IntConstant::Small(rhs)) => lhs.cmp(rhs),
//         }
//     }
// }

impl Sub for Constant {
    type Output = Result<Constant, RuntimeError>;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Constant::Int(lhs), Constant::Int(rhs)) => Ok(Constant::Int((lhs - rhs)?)),
            (lhs, rhs) => Err(RuntimeError::InvalidBinaryOperation(
                lhs,
                BinaryOperator::Minus,
                rhs,
            )),
        }
    }
}

impl PartialOrd for Constant {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Constant::Int(lhs), Constant::Int(rhs)) => Some(lhs.cmp(rhs)),

            (lhs, rhs) => {
                if lhs == rhs {
                    Some(Ordering::Equal)
                } else {
                    None
                }
            }
        }
    }
}

fn combine_with_operator(
    lhs: Constant,
    op: BinaryOperator,
    rhs: Constant,
) -> Result<Constant, RuntimeError> {
    match op {
        BinaryOperator::Plus => lhs + rhs,
        BinaryOperator::Times => lhs * rhs,
        BinaryOperator::Minus => lhs - rhs,
        BinaryOperator::Smaller => lhs
            .partial_cmp(&rhs)
            .ok_or(RuntimeError::InvalidBinaryOperation(lhs, op, rhs))
            .map(|ord| Constant::Bool(ord == Ordering::Less)),
    }
}

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("unexpected end of stream")]
    UnexpectedEndOfStream,
    #[error("variable '{0}' not defined")]
    NotDefinedVariable(Word),
    #[error("variable '{0}' already defined")]
    AlreadyDefinedVariable(Word),
    #[error("unexpected token '{0}' ({1})")]
    UnexpectedToken(Token, String),
    #[error("function '{0}' already defined")]
    AlreadyDefinedFunction(Word),
    #[error("function '{0}' not defined")]
    NotDefinedFunction(Word),
}

pub struct Tokens(Vec<Token>);

impl Tokens {
    fn get(&self) -> &Vec<Token> {
        &self.0
    }

    pub fn from_code(chars: &str, filename: String) -> Result<Self, SourceTokenizeError> {
        let mut vec = Vec::new();

        let mut p = Parsee::new(chars, filename);

        p.skip_whitespaces();

        while let Some(token) = p.parse_token()? {
            vec.push(token);
        }

        Ok(Self(vec))
    }
}

pub struct TokenStream<'a> {
    tokens: &'a Tokens,
    index: usize,
    #[expect(dead_code)]
    int_variables: IndexSet<Word>,
    //TODO pub
    pub expressions: Expressions,
    fun_names: IndexSet<Word>,
    stack: Stack,
}

impl<'a> TokenStream<'a> {
    pub fn new(tokens: &'a Tokens) -> Self {
        Self {
            tokens,
            index: 0,
            int_variables: IndexSet::new(),
            expressions: Expressions::new(),
            fun_names: IndexSet::new(),
            stack: Stack::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOperator {
    Plus,
    Times,
    Minus,
    Smaller,
}

impl Display for BinaryOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryOperator::Plus => write!(f, "+"),
            BinaryOperator::Times => write!(f, "*"),
            BinaryOperator::Minus => write!(f, "-"),
            BinaryOperator::Smaller => write!(f, "<"),
        }
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum Stickyness {
    Addition,
    Multiplication,
    Comparison,
}

impl BinaryOperator {
    fn stickyness(&self) -> Stickyness {
        match self {
            BinaryOperator::Plus => Stickyness::Addition,
            BinaryOperator::Times => Stickyness::Multiplication,
            BinaryOperator::Minus => Stickyness::Addition,
            BinaryOperator::Smaller => Stickyness::Comparison,
        }
    }
}

#[derive(Clone, Copy)]
enum FollowUp {
    BinaryOperator(BinaryOperator),
    End,
}

struct Stack {
    variables: IndexSet<Word>,
}
impl Stack {
    fn push(&mut self, word: Word) -> Result<VariableId, ParseError> {
        let (res, was_inserted) = self.variables.insert_full(word);

        if was_inserted {
            Ok(VariableId(res))
        } else {
            Err(ParseError::AlreadyDefinedVariable(
                self.variables
                    .get_index(res)
                    .expect("was not inserted")
                    .clone(),
            ))
        }
    }

    fn new() -> Self {
        Self {
            variables: IndexSet::new(),
        }
    }

    fn get(&self, word: &Word) -> Result<VariableId, ParseError> {
        self.variables
            .get_index_of(word)
            .ok_or(ParseError::NotDefinedVariable(word.clone()))
            .map(VariableId)
    }
}

impl TokenStream<'_> {
    pub fn is_fully_parsed(&self) -> bool {
        self.index == self.tokens.get().len()
    }

    fn peek(&mut self) -> Option<Token> {
        self.tokens.get().get(self.index).cloned()
    }

    fn next(&mut self) -> Result<Token, ParseError> {
        let token = self.peek().ok_or(ParseError::UnexpectedEndOfStream)?;
        self.index += 1;
        Ok(token)
    }

    fn parse_until_stickyness(
        &mut self,
        stickyness_threshold: Stickyness,
    ) -> Result<(Expr, FollowUp), ParseError> {
        let ast = self.parse_non_binary()?;

        let follow_up = self.parse_follow_up()?;

        if let FollowUp::BinaryOperator(op) = follow_up {
            let stickyness = op.stickyness();
            if stickyness > stickyness_threshold {
                let (rhs, follow_up) = self.parse_until_stickyness(stickyness_threshold)?;
                let expr = Expr::BinaryOp(self.expressions.add(ast), op, self.expressions.add(rhs));
                return Ok((expr, follow_up));
            }
        }

        Ok((ast, follow_up))
    }

    fn parse_non_binary(&mut self) -> Result<Expr, ParseError> {
        let token = self.next()?;

        match token {
            Token::ParanLeft => {
                let inner_expr = self.parse_expr()?;
                self.expect(Token::ParanRight)?;
                Ok(inner_expr)
            }
            Token::Return => {
                let expr = self.parse_expr()?;
                let expr = Expr::Return(self.expressions.add(expr));
                Ok(expr)
            }
            //TODO: Should we use self.int_variables to reduce this to an id here?
            Token::Word(word) => {
                if self.peek() == Some(Token::ParanLeft) {
                    let (exprs, _) = self.parse_expression_list_and_has_trailing_separator(
                        Token::ParanLeft,
                        Token::Comma,
                        Token::ParanRight,
                    )?;
                    let arguments = self.expressions.add_range(exprs);

                    let expr = Expr::Call(Call {
                        fun_id: FunId(self.fun_names.insert_full(word).0),
                        arguments,
                    });

                    Ok(expr)
                } else {
                    let expr = Expr::Variable(self.stack.get(&word)?);
                    Ok(expr)
                }
            }
            Token::IntConstant(int_constant) => {
                let expr = Expr::Constant(int_constant);
                Ok(expr)
            }
            _ => Err(ParseError::UnexpectedToken(
                token,
                "parse_non_binary".into(),
            )),
        }
    }

    fn parse_follow_up(&mut self) -> Result<FollowUp, ParseError> {
        match self.peek() {
            Some(Token::BinaryOperator(op)) => {
                self.next().expect("just peeked");
                Ok(FollowUp::BinaryOperator(op))
            }
            None
            | Some(Token::Semicolon | Token::BraceLeft | Token::ParanRight | Token::BraceRight) => {
                Ok(FollowUp::End)
            }
            Some(follow_up_token) => Err(ParseError::UnexpectedToken(
                follow_up_token.clone(),
                "parse_follow_up".into(),
            )),
        }
    }

    pub fn parse_if(&mut self) -> Result<Expr, ParseError> {
        self.expect(Token::If)?;

        let cond = self.parse_expr()?;

        //TODO: Check if boolean

        let if_block = self.parse_block()?;

        let expr = Expr::If(self.expressions.add(cond), if_block);
        Ok(expr)
    }

    pub fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        if let Some(Token::If) = self.peek() {
            return self.parse_if();
        }

        let mut ast = self.parse_non_binary()?;
        let mut follow_up = self.parse_follow_up()?;

        while let FollowUp::BinaryOperator(op) = follow_up {
            let (rhs, new_follow_up) = self.parse_until_stickyness(op.stickyness())?;
            let expr = Expr::BinaryOp(self.expressions.add(ast), op, self.expressions.add(rhs));
            ast = expr;
            follow_up = new_follow_up;
        }
        Ok(ast)
    }

    fn entertain(&mut self, entertained: Token) -> bool {
        if self.peek() == Some(entertained) {
            self.next().expect("just peeked");
            true
        } else {
            false
        }
    }

    fn expect(&mut self, expected: Token) -> Result<(), ParseError> {
        let actual = self.next()?;
        if expected == actual {
            Ok(())
        } else {
            Err(ParseError::UnexpectedToken(
                actual,
                format!("expected '{expected}'"),
            ))
        }
    }

    fn expect_word(&mut self) -> Result<Word, ParseError> {
        match self.next()? {
            Token::Word(word) => Ok(word),
            token => Err(ParseError::UnexpectedToken(token, "expected a word".into())),
        }
    }

    fn parse_parameters(&mut self) -> Result<Variables, ParseError> {
        self.expect(Token::ParanLeft)?;

        let mut variables = Variables::new();

        if self.entertain(Token::ParanRight) {
            return Ok(variables);
        }

        loop {
            let name = self.expect_word()?;
            self.expect(Token::Colon)?;
            let ty = self.expect_type()?;
            variables.insert_new(Variable::new(name, ty))?;

            if self.entertain(Token::ParanRight) {
                break;
            }

            self.expect(Token::Comma)?;
        }

        Ok(variables)
    }

    pub fn parse_fun(&mut self) -> Result<Fun, ParseError> {
        self.expect(Token::Fun)?;

        //TODO Meh.
        assert!(self.stack.variables.is_empty());

        let name = self.expect_word()?;

        let variables = self.parse_parameters()?;

        self.expect(Token::Arrow)?;
        let ty = self.expect_type()?;

        for variable in variables.set.keys() {
            //TODO Test error
            self.stack.push(variable.clone())?;
        }

        let body = self.parse_block()?;

        self.stack.variables.clear();

        Ok(Fun::new(name, variables, ty, body))
    }

    fn expect_type(&mut self) -> Result<Ty, ParseError> {
        match self.next()? {
            Token::Ty(word) => Ok(word),
            token => Err(ParseError::UnexpectedToken(token, "expected a type".into())),
        }
    }

    fn parse_expression_list_and_has_trailing_separator(
        &mut self,
        left_token: Token,
        separator_token: Token,
        right_token: Token,
    ) -> Result<(Vec<Expr>, bool), ParseError> {
        self.expect(left_token)?;

        let mut statements = Vec::new();

        let mut has_trailing_separator = false;

        loop {
            if self.entertain(right_token.clone()) {
                break;
            }

            if self.entertain(separator_token.clone()) {
                has_trailing_separator = true;
                continue;
            }

            statements.push(self.parse_expr()?);
            has_trailing_separator = false;
        }

        Ok((statements, has_trailing_separator))
    }

    fn parse_block(&mut self) -> Result<Block, ParseError> {
        let (mut statements, has_trailing_separator) = self
            .parse_expression_list_and_has_trailing_separator(
                Token::BraceLeft,
                Token::Semicolon,
                Token::BraceRight,
            )?;

        if has_trailing_separator {
            let expr = Expr::Constant(Constant::None);
            statements.push(expr);
        }

        let range = self.expressions.add_range(statements);

        Ok(Block::new(range))
    }
}

impl Tokens {
    pub fn parse(&self) -> Result<Ast, ParseError> {
        let mut fun_set = IndexMap::new();

        let mut ts = TokenStream::new(self);

        loop {
            match ts.peek() {
                None => break,
                Some(Token::Fun) => {
                    let fun: Fun = ts.parse_fun()?;
                    match fun_set.entry(fun.name.clone()) {
                        Entry::Occupied(_) => {
                            return Err(ParseError::AlreadyDefinedFunction(fun.name))
                        }
                        Entry::Vacant(vacant_entry) => {
                            ts.fun_names.insert(fun.name.clone());
                            vacant_entry.insert(fun);
                        }
                    }
                }
                Some(unexpected) => {
                    return Err(ParseError::UnexpectedToken(
                        unexpected,
                        "expected a top level declaration (currently only 'fun')".into(),
                    ))
                }
            }
        }

        // This should be implicit by the above code.
        assert!(ts.is_fully_parsed());

        let entry_point = ts.fun_names.get_index_of(&*MAIN).map(FunId);
        //TODO: Verify that there are no arguments for main?

        let mut funs = Vec::new();
        for name in ts.fun_names {
            funs.push(
                fun_set
                    .swap_remove(&name)
                    .ok_or(ParseError::NotDefinedFunction(name))?,
            );
        }

        Ok(Ast::new(funs, ts.expressions, entry_point))
    }
}

pub struct Expressions {
    vec: Vec<Expr>,
}
impl Expressions {
    fn new() -> Self {
        Self { vec: Vec::new() }
    }

    fn add(&mut self, expr: Expr) -> ExprId {
        let res = self.next_expr_id();
        self.vec.push(expr);
        res
    }

    fn get(&self, expr_id: ExprId) -> &Expr {
        &self.vec[expr_id.0]
    }

    //TODO private
    fn next_expr_id(&self) -> ExprId {
        ExprId(self.vec.len())
    }

    fn add_range(&mut self, exprs: impl IntoIterator<Item = Expr>) -> ExprRange {
        let start = self.next_expr_id();
        self.vec.extend(exprs);
        let end = self.next_expr_id();
        ExprRange { start, end }
    }
}

pub struct Ast {
    funs: Vec<Fun>,
    expressions: Expressions,
    entry_point: Option<FunId>,
}

// pub struct AnnotatedAst {
//     ast: Ast,
// }
// impl AnnotatedAst {
//     pub fn to_code(&self) -> String {
//         for fun in self.ast.funs {
//             self.to_code_fun(fun)
//         }
//     }

//     fn to_code_fun(&self, fun: Fun) {
//         todo!()
//     }
// }

impl Ast {
    fn evaluate_block(
        &self,
        block: &Block,
        variable_values: &VariableValues,
    ) -> Result<Evaluation, RuntimeError> {
        let Some(last_id) = block.statements.end.0.checked_sub(1) else {
            return Ok(Constant::None.into());
        };

        for id in block.statements.start.0..last_id {
            let expr_id = ExprId(id);
            let eval = self.evaluate_expr(expr_id, variable_values)?;
            let Some(_) = eval.some_or_please_return() else {
                return Ok(eval);
            };
        }

        self.evaluate_expr(ExprId(last_id), variable_values)
    }

    fn evaluate_expr(
        &self,
        expr_id: ExprId,
        variable_values: &VariableValues,
    ) -> Result<Evaluation, RuntimeError> {
        match self.expressions.get(expr_id) {
            Expr::Constant(int_constant) => Ok((*int_constant).into()),
            Expr::Variable(variable_id) => Ok(variable_values.get(variable_id)?.into()),
            Expr::BinaryOp(lhs_expr, op, rhs_expr) => {
                let lhs = self.evaluate_expr(*lhs_expr, variable_values)?;
                let Some(lhs) = lhs.some_or_please_return() else {
                    return Ok(lhs);
                };

                let rhs = self.evaluate_expr(*rhs_expr, variable_values)?;
                let Some(rhs) = rhs.some_or_please_return() else {
                    return Ok(rhs);
                };

                Ok(combine_with_operator(lhs, *op, rhs)?.into())
            }
            Expr::Block(block) => self.evaluate_block(block, variable_values),
            Expr::Return(expr) => self
                .evaluate_expr(*expr, variable_values)
                .map(|evaluation| evaluation.returning()),
            Expr::If(expr, block) => {
                let eval = self.evaluate_expr(*expr, variable_values)?;
                let Some(constant) = eval.some_or_please_return() else {
                    return Ok(eval);
                };

                let b = constant.get_bool()?;

                if b {
                    let block_result = self.evaluate_block(block, variable_values)?;
                    let Some(_) = block_result.some_or_please_return() else {
                        return Ok(block_result);
                    };
                }

                Ok(Constant::None.into())
            }
            Expr::Call(call) => {
                let mut parameters = Vec::new();
                for id in call.arguments.start.0..call.arguments.end.0 {
                    let eval = self.evaluate_expr(ExprId(id), variable_values)?;
                    let Some(parameter) = eval.some_or_please_return() else {
                        return Ok(eval);
                    };

                    parameters.push(parameter);
                }
                self.evaluate_fun(call.fun_id, parameters).map(|c| c.into())
            }
        }
    }

    pub fn evaluate_fun(
        &self,
        fun_id: FunId,
        parameters: Vec<Constant>,
    ) -> Result<Constant, RuntimeError> {
        let fun = self
            .funs
            .get(fun_id.0)
            .unwrap() //TODO
            // .ok_or_else(|| RuntimeError::UnknownFunction(name.clone()))?
            ;

        if fun.variables.len() != parameters.len() {
            return Err(RuntimeError::IncompatibleParameters(
                parameters,
                fun.variables.clone(),
            ));
        }

        let variable_values = VariableValues::new(parameters);

        let result = self
            .evaluate_block(&fun.body, &variable_values)?
            .unwrap_on_outer_layer();

        if result.ty() != fun.ty {
            return Err(RuntimeError::WrongReturnType(
                fun.name.clone(),
                fun.ty,
                result.ty(),
            ));
        }

        Ok(result)
    }

    pub fn evaluate_main(&self) -> Result<Constant, RuntimeError> {
        self.evaluate_fun(
            self.entry_point.ok_or(RuntimeError::NoEntryPoint)?,
            Vec::new(),
        )
    }

    fn new(funs: Vec<Fun>, expressions: Expressions, entry_point: Option<FunId>) -> Self {
        Self {
            funs,
            expressions,
            entry_point,
        }
    }
}

//TODO: Type deduce
pub fn evaluate_debug(
    expr: Expr,
    ty: Ty,
    mut expressions: Expressions,
) -> Result<Constant, RuntimeError> {
    let name = MAIN.clone();

    let expr_id = expressions.add(expr);

    let block = Block {
        statements: ExprRange {
            start: expr_id,
            end: ExprId(expr_id.0 + 1),
        },
    };

    let main = Fun::new(name.clone(), Variables::new(), ty, block);

    let ast = Ast::new(vec![main], expressions, Some(FunId(0)));

    ast.evaluate_main()
}

static MAIN: LazyLock<Word> = LazyLock::new(|| "main".try_into().expect("valid word"));

pub struct Fun {
    name: Word,
    variables: Variables,
    ty: Ty,
    body: Block,
}
impl Fun {
    fn new(name: Word, variables: Variables, ty: Ty, body: Block) -> Self {
        Self {
            name,
            variables,
            ty,
            body,
        }
    }
}

#[derive(Eq, Clone, Hash, PartialEq, Debug)]
struct Variable {
    name: Word,
    ty: Ty,
}
impl Variable {
    fn new(name: Word, ty: Ty) -> Self {
        Self { name, ty }
    }
}

#[derive(Debug, Error)]
pub enum TokenizeError {
    #[error("unknown token'{0}'")]
    UnknownToken(String),
    #[error("unknown character: '{0}'")]
    UnknownCharacter(char),

    #[error("invalid word: '{0}'")]
    InvalidWord(String),
}

struct Parsee {
    //Should not be empty.
    lines: Vec<Vec<char>>,
    current_line_id: usize,
    current_char_id: usize,
    filename: String,
}

impl Parsee {
    fn peek(&mut self) -> Option<char> {
        self.lines
            .get(self.current_line_id)
            .map(|line| line[self.current_char_id])
    }

    fn next(&mut self) -> Option<char> {
        let c = self.peek()?;

        self.current_char_id += 1;
        if self.lines[self.current_line_id].len() == self.current_char_id {
            self.current_line_id += 1;
            self.current_char_id = 0;
        }

        Some(c)
    }

    fn new(chars: &str, filename: String) -> Self {
        let lines = chars
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| line.chars().collect())
            .collect();
        Self {
            lines,
            current_line_id: 0,
            current_char_id: 0,
            filename,
        }
    }

    fn skip_whitespaces(&mut self) {
        //TODO
        while self.peek().is_some_and(|c| c.is_whitespace()) {
            self.next();
        }
    }

    fn parse_word(&mut self, start: char) -> Result<Token, SourceTokenizeError> {
        let mut word = String::new();
        word.push(start);

        while let Some(next) = self.peek() {
            if !next.is_ascii_alphanumeric() {
                break;
            }

            word.push(next);
            self.next();
        }

        if let Some(token) = KEYWORD_TOKEN_MAP.get(word.as_str()) {
            return Ok(token.clone());
        }

        Ok(Token::Word(word.try_into().map_err(|err| self.error(err))?))
    }

    fn parse_token(&mut self) -> Result<Option<Token>, SourceTokenizeError> {
        self.skip_whitespaces();

        let Some(start) = self.next() else {
            return Ok(None);
        };

        if let Some(token) = SINGLE_DIGIT_TOKEN_MAP.get(&start) {
            if matches!(token, Token::BinaryOperator(BinaryOperator::Minus))
                && Some('>') == self.peek()
            {
                self.next();
                return Ok(Some(Token::Arrow));
            }

            return Ok(Some(token.clone()));
        }

        let token = match start {
            '0'..='9' => self.parse_number(start)?,
            'A'..='z' => self.parse_word(start)?,
            _ => return Err(self.error(TokenizeError::UnknownCharacter(start))),
        };

        Ok(Some(token))
    }

    fn parse_number(&mut self, start: char) -> Result<Token, SourceTokenizeError> {
        let mut word = String::new();
        word.push(start);

        while let Some(next) = self.peek() {
            if !next.is_ascii_digit() {
                break;
            }

            word.push(next);
            self.next();
        }

        let i = word.parse().unwrap();

        Ok(Token::IntConstant(Constant::Int(IntConstant::Small(i))))
    }

    fn error(&self, error: TokenizeError) -> SourceTokenizeError {
        SourceTokenizeError::new(
            self.filename.clone(),
            self.current_line_id,
            self.current_char_id,
            self.current_line(),
            error,
        )
    }

    fn current_line(&self) -> String {
        self.lines
            .get(self.current_line_id)
            .expect("hopefully only called when still pointing to something")
            .iter()
            .collect()
    }
}

#[derive(Debug, Error)]
pub struct SourceError<E: Error> {
    line_id: usize,
    character_id: usize,
    filename: String,
    line: String,
    error: E,
}

impl<E: Display + Error> Display for SourceError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut pointer = String::new();
        for _ in 0..self.character_id - 1 {
            pointer.push(' ');
        }
        pointer.push('^');

        write!(
            f,
            "{}:{}:{}\n{}\n{}\n{}",
            self.filename,
            self.line_id + 1,
            self.character_id,
            self.line,
            pointer,
            self.error
        )
    }
}

pub type SourceTokenizeError = SourceError<TokenizeError>;

impl<E: Error> SourceError<E> {
    fn new(filename: String, line_id: usize, character_id: usize, line: String, error: E) -> Self {
        Self {
            filename,
            line_id,
            character_id,
            line,
            error,
        }
    }
}

static SINGLE_DIGIT_TOKEN_MAP: LazyLock<IndexMap<char, Token>> = std::sync::LazyLock::new(|| {
    indexmap! {
        ',' => Token::Comma,
        ';' => Token::Semicolon,
        '+' => Token::BinaryOperator(BinaryOperator::Plus),
        '-' => Token::BinaryOperator(BinaryOperator::Minus),
        '*' => Token::BinaryOperator(BinaryOperator::Times),
        '<' => Token::BinaryOperator(BinaryOperator::Smaller),
        '(' => Token::ParanLeft,
        ')' => Token::ParanRight,
        '{' => Token::BraceLeft,
        '}' => Token::BraceRight,
        ':' => Token::Colon,

    }
});

static KEYWORD_TOKEN_MAP: LazyLock<IndexMap<&str, Token>> = std::sync::LazyLock::new(|| {
    indexmap! {
        "fun" => Token::Fun,
        "return" => Token::Return,
        "int" => Token::Ty(Ty::Int),
        "if" => Token::If,
        "call" => Token::Call,
    }
});

//TODO pub
pub mod word {
    use std::fmt::Display;

    use super::TokenizeError;

    #[derive(Hash, PartialEq, Eq, Debug, Clone)]
    pub struct Word(String);

    impl TryFrom<String> for Word {
        type Error = TokenizeError;

        fn try_from(value: String) -> Result<Self, Self::Error> {
            let mut chars = value.chars();

            let Some(first_char) = chars.next() else {
                return Err(Self::Error::InvalidWord(value));
            };

            if !first_char.is_ascii_lowercase() {
                return Err(Self::Error::InvalidWord(value));
            }

            if !chars.all(|c| c.is_ascii_alphanumeric()) {
                return Err(Self::Error::InvalidWord(value));
            }

            Ok(Self(value))
        }
    }

    impl TryFrom<&str> for Word {
        type Error = TokenizeError;

        fn try_from(value: &str) -> Result<Self, Self::Error> {
            value.to_string().try_into()
        }
    }

    impl Display for Word {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Copy)]
pub enum Ty {
    Int,
    Bool,
    None,
}

impl Display for Ty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Ty::Int => write!(f, "int"),
            Ty::Bool => write!(f, "bool"),
            Ty::None => write!(f, "()"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Token {
    Return,
    Ty(Ty),
    Word(Word),
    IntConstant(Constant),
    BinaryOperator(BinaryOperator),
    Semicolon,
    Colon,
    Arrow,
    Comma,
    ParanLeft,
    ParanRight,
    BraceLeft,
    BraceRight,
    Fun,
    If,
    Call,
}

impl Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::Return => write!(f, "return"),
            Token::Ty(ty) => write!(f, "{ty}"),
            Token::Word(word) => write!(f, "{word}"),
            Token::IntConstant(int_constant) => write!(f, "{int_constant}"),
            Token::BinaryOperator(binary_operator) => write!(f, "{binary_operator}"),
            Token::Semicolon => write!(f, ";"),
            Token::Colon => write!(f, ":"),
            Token::Arrow => write!(f, "->"),
            Token::Comma => write!(f, ","),
            Token::ParanLeft => write!(f, "("),
            Token::ParanRight => write!(f, ")"),
            Token::BraceLeft => write!(f, "{{"),
            Token::BraceRight => write!(f, "}}"),
            Token::Fun => write!(f, "fun"),
            Token::If => write!(f, "if"),
            Token::Call => write!(f, "call"),
        }
    }
}
