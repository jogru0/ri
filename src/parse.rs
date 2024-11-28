use std::{
    iter::Peekable,
    ops::{Add, Mul, Sub},
    str::Chars,
};

use indexmap::IndexSet;

#[derive(Debug)]
pub enum RuntimError {
    NotImplememted,
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
                .map(|val| Self::Small(val)),
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
                .map(|val| Self::Small(val)),
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
                .map(|val| Self::Small(val)),
        }
    }
}

pub enum Expr {
    Constant(IntConstant),
    Variable(usize),
    Sum(Box<Expr>, BinaryOperator, Box<Expr>),
    Block(Block),
}

pub struct Block {}

struct Variables {
    ints: Vec<IntConstant>,
}

fn evaluate_with_variables(expr: Expr, variables: &Variables) -> Result<IntConstant, RuntimError> {
    match expr {
        Expr::Constant(int_constant) => Ok(int_constant),
        Expr::Variable(i) => Ok(variables.ints[i]),
        Expr::Sum(lhs_expr, op, rhs_expr) => {
            let lhs = evaluate_with_variables(*lhs_expr, variables)?;
            let rhs = evaluate_with_variables(*rhs_expr, variables)?;
            combine_with_operator(lhs, op, rhs)
        }
        Expr::Block(_) => todo!(),
    }
}

pub fn evaluate(expr: Expr) -> Result<IntConstant, RuntimError> {
    evaluate_with_variables(expr, &Variables { ints: Vec::new() })
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
    UndefinedVariable(String),
    UnexpectedToken(Token),
}

pub struct TokenStream {
    tokens: Vec<Token>,
    index: usize,
    int_variables: IndexSet<String>,
}

#[derive(Clone, Copy, Debug)]
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

    fn next(&mut self) -> Result<Token, ParseError> {
        let token = self
            .tokens
            .get(self.index)
            .ok_or(ParseError::UnexpectedEndOfStream)?;

        self.index += 1;
        Ok(token.clone())
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
                return Ok((Expr::Sum(Box::new(ast), op, Box::new(rhs)), follow_up));
            }
        }

        Ok((ast, follow_up))
    }

    fn parse_non_binary(&mut self) -> Result<Expr, ParseError> {
        let token = self.next()?;
        match token {
            Token::Return => todo!(),
            Token::IntVariable(s) => {
                let id = self
                    .int_variables
                    .get_index_of(&s)
                    .ok_or(ParseError::UndefinedVariable(s))?;
                Ok(Expr::Variable(id))
            }
            Token::IntConstant(int_constant) => Ok(Expr::Constant(int_constant)),
            Token::BinaryOperator(_) | Token::Semicolon | Token::End => {
                Err(ParseError::UnexpectedToken(token))
            }
        }
    }

    fn parse_follow_up(&mut self) -> Result<FollowUp, ParseError> {
        let follow_up_token = self.next()?;

        match follow_up_token {
            Token::Return | Token::IntVariable(..) | Token::IntConstant(..) => {
                Err(ParseError::UnexpectedToken(follow_up_token))
            }
            Token::BinaryOperator(op) => Ok(FollowUp::BinaryOperator(op)),
            Token::Semicolon => Ok(FollowUp::End),
            Token::End => todo!(),
        }
    }

    pub fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        let mut ast = self.parse_non_binary()?;
        let mut follow_up = self.parse_follow_up()?;

        while let FollowUp::BinaryOperator(op) = follow_up {
            let (rhs, new_follow_up) = self.parse_until_stickyness(op.stickyness())?;
            ast = Expr::Sum(Box::new(ast), op, Box::new(rhs));
            follow_up = new_follow_up;
        }
        Ok(ast)
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

        let token = if word == "return" {
            Token::Return
        } else {
            Token::IntVariable(word)
        };

        Ok(token)
    }

    fn parse_token(&mut self) -> Result<Option<Token>, TokenizeError> {
        self.skip_whitespaces();

        let Some(start) = self.next() else {
            return Ok(None);
        };

        let token = match start {
            ';' => Token::Semicolon,
            '+' => Token::BinaryOperator(BinaryOperator::Plus),
            '-' => Token::BinaryOperator(BinaryOperator::Minus),
            '*' => Token::BinaryOperator(BinaryOperator::Times),
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

#[derive(Clone, Debug)]
pub enum Token {
    Return,
    IntVariable(String),
    IntConstant(IntConstant),
    BinaryOperator(BinaryOperator),
    Semicolon,
    End,
}
