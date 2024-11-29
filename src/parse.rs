use std::{
    iter::Peekable,
    ops::{Add, Mul, Sub},
    str::Chars,
    sync::LazyLock,
};

use indexmap::{indexmap, IndexMap, IndexSet};

#[derive(Debug)]
pub enum RuntimError {
    NotImplememted,
    UnknownVariable(Word),
    IncompatibleParameters(Word, IndexSet<Word>),
    MisingReturn,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntConstant {
    Small(i128),
}

impl Sub for IntConstant {
    type Output = Result<IntConstant, RuntimError>;

    fn sub(self, other: Self) -> Self::Output {
        match (self, other) {
            (IntConstant::Small(lhs), IntConstant::Small(rhs)) => lhs
                .checked_sub(rhs)
                .ok_or(RuntimError::NotImplememted)
                .map(Self::Small),
        }
    }
}

impl Mul for IntConstant {
    type Output = Result<IntConstant, RuntimError>;

    fn mul(self, other: Self) -> Self::Output {
        match (self, other) {
            (IntConstant::Small(lhs), IntConstant::Small(rhs)) => lhs
                .checked_mul(rhs)
                .ok_or(RuntimError::NotImplememted)
                .map(Self::Small),
        }
    }
}

impl Add for IntConstant {
    type Output = Result<IntConstant, RuntimError>;

    fn add(self, other: Self) -> Self::Output {
        match (self, other) {
            (IntConstant::Small(lhs), IntConstant::Small(rhs)) => lhs
                .checked_add(rhs)
                .ok_or(RuntimError::NotImplememted)
                .map(Self::Small),
        }
    }
}

pub enum Expr {
    Return(Box<Expr>),
    Constant(IntConstant),
    Variable(Word),
    BinaryOp(Box<Expr>, BinaryOperator, Box<Expr>),
    Block(Block),
}

pub struct Block {
    statements: Vec<Expr>,
}

impl Block {
    fn new(statements: Vec<Expr>) -> Self {
        Self { statements }
    }
}

struct Variables {
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
    word_to_value: IndexMap<Word, IntConstant>,
}

impl VariableValues {
    fn get(&self, word: Word) -> Result<IntConstant, RuntimError> {
        self.word_to_value
            .get(&word)
            .ok_or(RuntimError::UnknownVariable(word))
            .cloned()
    }

    pub fn new(word_to_value: IndexMap<Word, IntConstant>) -> Self {
        Self { word_to_value }
    }

    fn set(&self) -> IndexSet<Word> {
        self.word_to_value.keys().cloned().collect()
    }
}

#[must_use]
struct Evaluation {
    value: IntConstant,
    returning: bool,
}

impl Evaluation {
    fn returning(self) -> Self {
        Self {
            value: self.value,
            returning: true,
        }
    }

    fn unwrap_on_outer_layer(&self) -> IntConstant {
        self.value
    }

    fn some_or_please_return(&self) -> Option<IntConstant> {
        if self.returning {
            None
        } else {
            Some(self.value)
        }
    }
}

impl From<IntConstant> for Evaluation {
    fn from(val: IntConstant) -> Self {
        Evaluation {
            value: val,
            returning: false,
        }
    }
}

fn evaluate_with_variables(
    expr: Expr,
    variable_values: &VariableValues,
) -> Result<Evaluation, RuntimError> {
    match expr {
        Expr::Constant(int_constant) => Ok(int_constant.into()),
        Expr::Variable(word) => Ok(variable_values.get(word)?.into()),
        Expr::BinaryOp(lhs_expr, op, rhs_expr) => {
            let lhs = evaluate_with_variables(*lhs_expr, variable_values)?;
            let Some(lhs) = lhs.some_or_please_return() else {
                return Ok(lhs);
            };

            let rhs = evaluate_with_variables(*rhs_expr, variable_values)?;
            let Some(rhs) = rhs.some_or_please_return() else {
                return Ok(rhs);
            };

            Ok(combine_with_operator(lhs, op, rhs)?.into())
        }
        Expr::Block(Block { statements }) => {
            for statement in statements {
                let eval = evaluate_with_variables(statement, variable_values)?;
                let Some(_) = eval.some_or_please_return() else {
                    return Ok(eval);
                };
            }

            Err(RuntimError::MisingReturn)
        }
        Expr::Return(expr) => {
            evaluate_with_variables(*expr, variable_values).map(|evaluation| evaluation.returning())
        }
    }
}

pub fn evaluate(expr: Expr) -> Result<IntConstant, RuntimError> {
    Ok(
        evaluate_with_variables(expr, &VariableValues::new(IndexMap::new()))?
            .unwrap_on_outer_layer(),
    )
}

pub fn evaluate_fun(fun: Fun, variable_values: VariableValues) -> Result<IntConstant, RuntimError> {
    let variable_values_set = variable_values.set();
    if fun.variables.set() != variable_values_set {
        return Err(RuntimError::IncompatibleParameters(
            fun.name,
            variable_values_set,
        ));
    }

    Ok(evaluate_with_variables(Expr::Block(fun.body), &variable_values)?.unwrap_on_outer_layer())
}

fn combine_with_operator(
    lhs: IntConstant,
    op: BinaryOperator,
    rhs: IntConstant,
) -> Result<IntConstant, RuntimError> {
    match op {
        BinaryOperator::Plus => lhs + rhs,
        BinaryOperator::Times => lhs * rhs,
        BinaryOperator::Minus => lhs - rhs,
    }
}

#[derive(Debug)]
pub enum ParseError {
    UnexpectedEndOfStream,
    UndefinedVariable(Word),
    AlreadyDefinedVariable(Word),
    UnexpectedToken(Token),
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
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum Stickyness {
    Addition,
    Multiplication,
}

impl BinaryOperator {
    fn stickyness(&self) -> Stickyness {
        match self {
            BinaryOperator::Plus => Stickyness::Addition,
            BinaryOperator::Times => Stickyness::Multiplication,
            BinaryOperator::Minus => Stickyness::Addition,
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

    fn peek(&mut self) -> Result<Token, ParseError> {
        self.tokens
            .get(self.index)
            .ok_or(ParseError::UnexpectedEndOfStream)
            .cloned()
    }

    fn next(&mut self) -> Result<Token, ParseError> {
        let token = self.peek()?;
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
            Token::Return => Ok(Expr::Return(Box::new(self.parse_expr()?))),
            //TODO: Should we use self.int_variables to reduce this to an id here?
            Token::Word(s) => Ok(Expr::Variable(s)),
            Token::IntConstant(int_constant) => Ok(Expr::Constant(int_constant)),
            _ => Err(ParseError::UnexpectedToken(token)),
        }
    }

    fn parse_follow_up(&mut self) -> Result<FollowUp, ParseError> {
        match self.peek() {
            Ok(Token::BinaryOperator(op)) => {
                self.next().expect("just peeked");
                Ok(FollowUp::BinaryOperator(op))
            }
            //TODO kind of error flow
            Err(ParseError::UnexpectedEndOfStream) | Ok(Token::Semicolon) => Ok(FollowUp::End),
            Ok(follow_up_token) => Err(ParseError::UnexpectedToken(follow_up_token)),
            Err(err) => unreachable!("unexpected error '{err:?}'"),
        }
    }

    pub fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        let mut ast = self.parse_non_binary()?;
        let mut follow_up = self.parse_follow_up()?;

        while let FollowUp::BinaryOperator(op) = follow_up {
            let (rhs, new_follow_up) = self.parse_until_stickyness(op.stickyness())?;
            ast = Expr::BinaryOp(Box::new(ast), op, Box::new(rhs));
            follow_up = new_follow_up;
        }
        Ok(ast)
    }

    fn entertain(&mut self, entertained: Token) -> Result<bool, ParseError> {
        if self.peek()? == entertained {
            self.next().expect("just peeked");
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn expect(&mut self, expected: Token) -> Result<(), ParseError> {
        let actual = self.next()?;
        if expected == actual {
            Ok(())
        } else {
            Err(ParseError::UnexpectedToken(actual))
        }
    }

    fn expect_word(&mut self) -> Result<Word, ParseError> {
        match self.next()? {
            Token::Word(word) => Ok(word),
            token => Err(ParseError::UnexpectedToken(token)),
        }
    }

    fn parse_parameters(&mut self) -> Result<Variables, ParseError> {
        self.expect(Token::ParanLeft)?;

        let mut variables = Variables::new();

        if self.entertain(Token::ParanRight)? {
            return Ok(variables);
        }

        loop {
            let name = self.expect_word()?;
            self.expect(Token::Colon)?;
            let ty = self.expect_type()?;
            variables.insert_new(Variable::new(name, ty))?;

            if self.entertain(Token::ParanRight)? {
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
            token => Err(ParseError::UnexpectedToken(token)),
        }
    }

    fn parse_block(&mut self) -> Result<Block, ParseError> {
        let _ = self.expect(Token::BraceLeft);

        let mut statements = Vec::new();

        loop {
            if self.entertain(Token::BraceRight)? {
                break;
            }

            statements.push(self.parse_expr()?);

            self.expect(Token::Semicolon)?;
        }

        Ok(Block::new(statements))
    }
}

pub struct Fun {
    name: Word,
    variables: Variables,
    #[expect(dead_code)]
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

#[derive(Eq, Hash, PartialEq)]
struct Variable {
    name: Word,
    ty: Ty,
}
impl Variable {
    fn new(name: Word, ty: Ty) -> Self {
        Self { name, ty }
    }
}

#[derive(Debug)]
pub enum TokenizeError {
    UnknownToken(String),
    UnknownCharacter(char),
}

pub fn parse_frfr(chars: &str) -> Result<TokenStream, TokenizeError> {
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

        Ok(Token::Word(Word(word)))
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

        Ok(Token::IntConstant(IntConstant::Small(i)))
    }
}

static SINGLE_DIGIT_TOKEN_MAP: LazyLock<IndexMap<char, Token>> = std::sync::LazyLock::new(|| {
    indexmap! {
        ',' => Token::Comma,
        ';' => Token::Semicolon,
        '+' => Token::BinaryOperator(BinaryOperator::Plus),
        '-' => Token::BinaryOperator(BinaryOperator::Minus),
        '*' => Token::BinaryOperator(BinaryOperator::Times),
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
    }
});

pub fn to_tokens(s: &str) -> Result<TokenStream, TokenizeError> {
    parse_frfr(s)
    // let vec: Vec<Token> = s
    //     .split_ascii_whitespace()
    //     .map(to_token)
    //     .collect::<Result<Vec<_>, _>>()?;

    // Ok(TokenStream {
    //     tokens: vec,
    //     index: 0,
    //     int_variables: IndexSet::new(),
    // })
}

#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct Word(String);

impl Word {
    pub fn new(value: String) -> Self {
        //TODO Check validity
        Self(value)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Ty {
    Int,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Token {
    Return,
    Ty(Ty),
    Word(Word),
    IntConstant(IntConstant),
    BinaryOperator(BinaryOperator),
    Semicolon,
    Colon,
    Arrow,
    End,
    Comma,
    ParanLeft,
    ParanRight,
    BraceLeft,
    BraceRight,
    Fun,
}
