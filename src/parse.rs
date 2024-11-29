use std::{
    cmp::Ordering,
    fmt::Display,
    iter::Peekable,
    ops::{Add, Mul, Sub},
    str::Chars,
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

pub enum Expr {
    Return(Box<Expr>),
    Constant(Constant),
    Variable(Word),
    BinaryOp(Box<Expr>, BinaryOperator, Box<Expr>),
    Block(Block),
    //TODO: Allow stuff like block here?
    If(Box<Expr>, Block),
    Call(Call),
}

pub struct Block {
    statements: Vec<Expr>,
}

pub struct Call {
    name: Word,
    arguments: Vec<Expr>,
}

impl Block {
    fn new(statements: Vec<Expr>) -> Self {
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

    fn set(&self) -> IndexSet<Word> {
        self.set.keys().cloned().collect()
    }
}

pub struct VariableValues {
    word_to_value: IndexMap<Word, Constant>,
}

impl VariableValues {
    fn get(&self, word: &Word) -> Result<Constant, RuntimeError> {
        self.word_to_value
            .get(word)
            .ok_or_else(|| RuntimeError::UnknownVariable(word.clone()))
            .cloned()
    }

    pub fn new(word_to_value: IndexMap<Word, Constant>) -> Self {
        Self { word_to_value }
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
    UndefinedVariable(Word),
    #[error("variable '{0}' already defined")]
    AlreadyDefinedVariable(Word),
    #[error("unexpected token '{0}' ({1})")]
    UnexpectedToken(Token, String),
    #[error("function '{0}' already defined")]
    AlreadyDefinedFunction(Word),
}

pub struct TokenStream {
    tokens: Vec<Token>,
    index: usize,
    #[expect(dead_code)]
    int_variables: IndexSet<Word>,
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

impl TokenStream {
    pub fn is_fully_parsed(&self) -> bool {
        self.index == self.tokens.len()
    }

    fn peek(&mut self) -> Option<Token> {
        self.tokens.get(self.index).cloned()
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
                return Ok((Expr::BinaryOp(Box::new(ast), op, Box::new(rhs)), follow_up));
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
            Token::Return => Ok(Expr::Return(Box::new(self.parse_expr()?))),
            //TODO: Should we use self.int_variables to reduce this to an id here?
            Token::Word(word) => {
                if self.peek() == Some(Token::ParanLeft) {
                    Ok(Expr::Call(Call {
                        name: word,
                        arguments: self.parse_expression_list(
                            Token::ParanLeft,
                            Token::Comma,
                            Token::ParanRight,
                        )?,
                    }))
                } else {
                    Ok(Expr::Variable(word))
                }
            }
            Token::IntConstant(int_constant) => Ok(Expr::Constant(int_constant)),
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

        Ok(Expr::If(Box::new(cond), if_block))
    }

    pub fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        if let Some(Token::If) = self.peek() {
            return self.parse_if();
        }

        let mut ast = self.parse_non_binary()?;
        let mut follow_up = self.parse_follow_up()?;

        while let FollowUp::BinaryOperator(op) = follow_up {
            let (rhs, new_follow_up) = self.parse_until_stickyness(op.stickyness())?;
            ast = Expr::BinaryOp(Box::new(ast), op, Box::new(rhs));
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

        let name = self.expect_word()?;

        let variables = self.parse_parameters()?;

        self.expect(Token::Arrow)?;
        let ty = self.expect_type()?;

        let body = self.parse_block()?;

        Ok(Fun::new(name, variables, ty, body))
    }

    fn expect_type(&mut self) -> Result<Ty, ParseError> {
        match self.next()? {
            Token::Ty(word) => Ok(word),
            token => Err(ParseError::UnexpectedToken(token, "expected a type".into())),
        }
    }

    fn parse_expression_list(
        &mut self,
        left_token: Token,
        separator_token: Token,
        right_token: Token,
    ) -> Result<Vec<Expr>, ParseError> {
        self.expect(left_token)?;

        let mut statements = Vec::new();

        loop {
            if self.entertain(right_token.clone()) {
                break;
            }

            if self.entertain(separator_token.clone()) {
                continue;
            }

            statements.push(self.parse_expr()?);
        }

        Ok(statements)
    }

    fn parse_block(&mut self) -> Result<Block, ParseError> {
        let _ = self.expect(Token::BraceLeft);

        let mut statements = Vec::new();

        loop {
            if self.entertain(Token::BraceRight) {
                break;
            }

            if self.entertain(Token::Semicolon) {
                continue;
            }

            statements.push(self.parse_expr()?);
        }

        Ok(Block::new(statements))
    }

    pub fn parse(mut self) -> Result<Ast, ParseError> {
        let mut funs = IndexMap::new();

        loop {
            match self.peek() {
                None => break,
                Some(Token::Fun) => {
                    let fun = self.parse_fun()?;
                    match funs.entry(fun.name.clone()) {
                        Entry::Occupied(_) => {
                            return Err(ParseError::AlreadyDefinedFunction(fun.name))
                        }
                        Entry::Vacant(vacant_entry) => {
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
        assert!(self.is_fully_parsed());

        Ok(Ast::new(funs))
    }
}

pub struct Ast {
    funs: IndexMap<Word, Fun>,
}

impl Ast {
    fn new(funs: IndexMap<Word, Fun>) -> Self {
        Self { funs }
    }

    fn evaluate_block(
        &self,
        block: &Block,
        variable_values: &VariableValues,
    ) -> Result<Evaluation, RuntimeError> {
        if block.statements.is_empty() {
            return Ok(Constant::None.into());
        }

        for statement in
            &block.statements[..block.statements.len().checked_sub(1).expect("not empty")]
        {
            let eval = self.evaluate_expr(statement, variable_values)?;
            let Some(_) = eval.some_or_please_return() else {
                return Ok(eval);
            };
        }

        self.evaluate_expr(block.statements.last().expect("not empty"), variable_values)
    }

    fn evaluate_expr(
        &self,
        expr: &Expr,
        variable_values: &VariableValues,
    ) -> Result<Evaluation, RuntimeError> {
        match expr {
            Expr::Constant(int_constant) => Ok((*int_constant).into()),
            Expr::Variable(word) => Ok(variable_values.get(word)?.into()),
            Expr::BinaryOp(lhs_expr, op, rhs_expr) => {
                let lhs = self.evaluate_expr(lhs_expr, variable_values)?;
                let Some(lhs) = lhs.some_or_please_return() else {
                    return Ok(lhs);
                };

                let rhs = self.evaluate_expr(rhs_expr, variable_values)?;
                let Some(rhs) = rhs.some_or_please_return() else {
                    return Ok(rhs);
                };

                Ok(combine_with_operator(lhs, *op, rhs)?.into())
            }
            Expr::Block(block) => self.evaluate_block(block, variable_values),
            Expr::Return(expr) => self
                .evaluate_expr(expr, variable_values)
                .map(|evaluation| evaluation.returning()),
            Expr::If(expr, block) => {
                let eval = self.evaluate_expr(expr, variable_values)?;
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
                for argument in &call.arguments {
                    let eval = self.evaluate_expr(argument, variable_values)?;
                    let Some(parameter) = eval.some_or_please_return() else {
                        return Ok(eval);
                    };

                    parameters.push(parameter);
                }
                self.evaluate_fun(&call.name, parameters).map(|c| c.into())
            }
        }
    }

    pub fn evaluate_fun(
        &self,
        name: &Word,
        parameters: Vec<Constant>,
    ) -> Result<Constant, RuntimeError> {
        let fun = self
            .funs
            .get(name)
            .ok_or_else(|| RuntimeError::UnknownFunction(name.clone()))?;

        let mut word_to_value = IndexMap::new();

        let variable_names = fun.variables.set();

        if variable_names.len() != parameters.len() {
            return Err(RuntimeError::IncompatibleParameters(
                parameters,
                fun.variables.clone(),
            ));
        }

        for (word, constant) in fun.variables.set().into_iter().zip(parameters) {
            let old = word_to_value.insert(word, constant);
            assert!(old.is_none());
        }

        let variable_values = VariableValues::new(word_to_value);

        Ok(self
            .evaluate_block(&fun.body, &variable_values)?
            .unwrap_on_outer_layer())
    }

    pub fn evaluate_main(&self) -> Result<Constant, RuntimeError> {
        self.evaluate_fun(&MAIN, Vec::new())
    }
}

//TODO: Type deduce
pub fn evaluate_debug(expr: Expr, ty: Ty) -> Result<Constant, RuntimeError> {
    let name = MAIN.clone();

    let main = Fun::new(
        name.clone(),
        Variables::new(),
        ty,
        Block {
            statements: vec![expr],
        },
    );

    let ast = Ast::new(indexmap! { name => main});

    ast.evaluate_main()
}

static MAIN: LazyLock<Word> = LazyLock::new(|| "main".try_into().expect("valid word"));

pub struct Fun {
    name: Word,
    variables: Variables,
    #[expect(dead_code)] //TODO
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

pub fn tokenize(chars: &str) -> Result<TokenStream, TokenizeError> {
    let mut vec = Vec::new();

    let mut p = Parsee::new(chars);

    p.skip_whitespaces();

    while let Some(token) = p.parse_token()? {
        vec.push(token);
    }

    Ok(TokenStream {
        tokens: vec,
        index: 0,
        int_variables: IndexSet::new(),
    })
}

struct Parsee<'a> {
    iter: Peekable<Chars<'a>>,
}

impl<'a> Parsee<'a> {
    fn peek(&mut self) -> Option<char> {
        self.iter.peek().copied()
    }

    fn next(&mut self) -> Option<char> {
        self.iter.next()
    }

    fn new(chars: &'a str) -> Self {
        Self {
            iter: chars.chars().peekable(),
        }
    }

    fn skip_whitespaces(&mut self) {
        //TODO
        while self.peek().is_some_and(|c| c.is_whitespace()) {
            self.next();
        }
    }

    fn parse_word(&mut self, start: char) -> Result<Token, TokenizeError> {
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

        Ok(Token::Word(word.try_into()?))
    }

    fn parse_token(&mut self) -> Result<Option<Token>, TokenizeError> {
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
            _ => return Err(TokenizeError::UnknownCharacter(start)),
        };

        Ok(Some(token))
    }

    fn parse_number(&mut self, start: char) -> Result<Token, TokenizeError> {
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
