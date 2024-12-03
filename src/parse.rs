use std::{
    cell::RefCell,
    cmp::Ordering,
    error::Error,
    fmt::Display,
    fs::read_to_string,
    mem::swap,
    ops::{Add, Mul, Rem, Sub},
    rc::Rc,
    sync::LazyLock,
};

use indexmap::{indexmap, map::Entry, IndexMap, IndexSet};
use itertools::Itertools;
use log::{debug, info};
use thiserror::Error;
use word::Word;

#[derive(Debug, Error)]

//TODO: Nicer List printing
pub enum RuntimeError {
    #[error("not (yet) implemented: {0}")]
    NotImplememted(String),
    //TODO: Obviously not so nice without the name. But it should never happen, right?
    #[error("unknown variable '{0:?}'")]
    UnknownVariable(VariableId),
    #[error("for function '{2}': arguments '{0:?}' incompatible with parameters '{1:?}'")]
    IncompatibleParameters(Vec<Ty>, Variables, Word),
    #[error("arguments '{0:?}' incompatible with internal function {1}")]
    IncompatibleParametersForInternal(Vec<Ty>, InternalFun),
    #[error("missing '{}'", Token::Return)]
    MisingReturn,
    #[error("unknown function '{0}'")]
    UnknownDotFunction(Word),
    //TODO constuctor, make sure they are unequal
    #[error("type error: expected '{0}', found '{1}'")]
    TypeError(Ty, Ty),
    #[error("invalid operation:'{0} {1} {2}'")]
    InvalidBinaryOperation(Constant, BinaryOperator, Constant),
    #[error("function '{0}' has return type '{1}', but evaluated to '{2}'")]
    WrongReturnType(Word, Ty, Ty),
    #[error("missing entry point (function '{}')", *MAIN)]
    NoEntryPoint,
    #[error("index '{1}' out of bounds for list {0:?})")]
    IndexOutOfBounds(Vec<Constant>, IntConstant),
    #[error("static function '{0}.{1}' not defined)")]
    StaticFunctionOnWrongType(Ty, InternalFun),
    #[error("operation '{0}' invalid on '{1:?}'")]
    InternalOperationInvalid(InternalFun, Vec<Constant>),
    #[error("encountered problem accessing file '{1}': {0}")]
    Io(std::io::Error, String),
    #[error("cannot parse '{0}' as '{1}': {2}")]
    Parse(String, Ty, std::num::ParseIntError),
    #[error("cannot iterate over '{0}'")]
    NotIterable(Constant),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum IntConstant {
    Small(i128),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Constant {
    Int(IntConstant),
    Bool(bool),
    None,
    List(Rc<RefCell<Vec<Constant>>>),
    Ty(Ty),
    Char(char),
    Callable(Callable),
}

impl From<Vec<Constant>> for Constant {
    fn from(value: Vec<Constant>) -> Self {
        Self::List(Rc::new(RefCell::new(value)))
    }
}

impl Constant {
    fn get_bool(&self) -> Result<bool, RuntimeError> {
        if let Constant::Bool(b) = self {
            Ok(*b)
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
            Constant::List(_) => Ty::List,
            Constant::Ty(_) => Ty::Ty,
            Constant::Char(_) => Ty::Char,
            Constant::Callable(_) => Ty::Callable,
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
            //TODO make it good
            Constant::List(vec) => write!(f, "{vec:?}"),
            Constant::Ty(ty) => write!(f, "{ty}"),
            Constant::Char(c) => write!(f, "{c}"),
            Constant::Callable(callable) => write!(f, "{callable}"),
        }
    }
}

impl Rem for IntConstant {
    type Output = Result<IntConstant, RuntimeError>;

    fn rem(self, other: Self) -> Self::Output {
        match (self, other) {
            (IntConstant::Small(lhs), IntConstant::Small(rhs)) => lhs
                .checked_rem(rhs)
                .ok_or(RuntimeError::NotImplememted("numbers beyond i128".into()))
                .map(Self::Small),
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

#[derive(Clone)]
pub enum Expr {
    Return(ExprId),
    Constant(Constant),
    Variable(VariableId),
    BinaryOp(ExprId, BinaryOperator, ExprId),
    Block(Block),
    //TODO: Test for block here
    If(ExprId, Block, ExprId),
    While(ExprId, Block),
    Call(Call),
    Assign(VariableId, ExprId),
    Introduce(VariableId, ExprId),
    For(VariableId, Option<VariableId>, ExprId, Block),
    Negate(ExprId),
}

#[derive(Clone, Copy, Debug)]
pub struct VariableId(usize);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct ExprId(usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ListId(usize);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
//TODO pub field
pub struct FunId(pub usize);

impl Display for FunId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ExprRange {
    start: ExprId,
    end: ExprId,
}

#[derive(Clone, Copy, Debug)]
pub struct Block {
    statements: ExprRange,
}

#[derive(Clone)]
pub struct Call {
    callable: Callable,
    arguments: ExprRange,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Callable {
    Fun(FunId),
    InternalFun(InternalFun),
}

impl Display for Callable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Callable::Fun(fun_id) => writeln!(f, "{fun_id}"),
            Callable::InternalFun(internal_fun) => writeln!(f, "{internal_fun}"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InternalFun {
    New,
    Invoke,
    Debug,
    Push,
    Get,
    Pop,
    FromFile,
    Len,
    SplitWhitespace,
    Parse,
    Sort,
    Abs,
    Lines,
}

impl Display for InternalFun {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InternalFun::New => write!(f, "new"),
            InternalFun::Push => write!(f, "push"),
            InternalFun::Get => write!(f, "get"),
            InternalFun::Pop => write!(f, "pop"),
            InternalFun::FromFile => write!(f, "from_file"),
            InternalFun::Len => write!(f, "len"),
            InternalFun::SplitWhitespace => write!(f, "split_whitespace"),
            InternalFun::Parse => write!(f, "parse"),
            InternalFun::Sort => write!(f, "sort"),
            InternalFun::Abs => write!(f, "abs"),
            InternalFun::Lines => write!(f, "lines"),
            InternalFun::Debug => write!(f, "debug"),
            InternalFun::Invoke => write!(f, "invoke"),
        }
    }
}

static WORD_INTERNAL_FUN_MAP: LazyLock<IndexMap<Word, InternalFun>> = LazyLock::new(|| {
    indexmap! {
        "new".try_into().expect("const") => InternalFun::New,
        "push".try_into().expect("const") => InternalFun::Push,
        "get".try_into().expect("const") => InternalFun::Get,
        "pop".try_into().expect("const") => InternalFun::Pop,
        "from_file".try_into().expect("const") => InternalFun::FromFile,
        "len".try_into().expect("const") => InternalFun::Len,
        "split_whitespace".try_into().expect("const") => InternalFun::SplitWhitespace,
        "parse".try_into().expect("const") => InternalFun::Parse,
        "sort".try_into().expect("const") => InternalFun::Sort,
        "abs".try_into().expect("const") => InternalFun::Abs,
        "lines".try_into().expect("const") => InternalFun::Lines,
        "debug".try_into().expect("const") => InternalFun::Debug,
        "invoke".try_into().expect("const") => InternalFun::Invoke,
    }
});

impl InternalFun {
    fn get(name: &Word) -> Option<Self> {
        WORD_INTERNAL_FUN_MAP.get(name).copied()
    }
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
    reference: usize,
}

impl VariableValues {
    fn get(&self, variable_id: &VariableId) -> Result<Constant, RuntimeError> {
        self.values
            .get(variable_id.0 + self.reference)
            .ok_or_else(|| //RuntimeError::UnknownVariable(word.clone())
                panic!("how"))
            .cloned()
    }

    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            reference: 0,
        }
    }

    fn set(&mut self, variable_id: VariableId, mut new: Constant) -> Result<(), RuntimeError> {
        let old = self
            .values
            .get_mut(variable_id.0 + self.reference)
            .ok_or(RuntimeError::UnknownVariable(variable_id))?;

        if old.ty() != new.ty() {
            return Err(RuntimeError::TypeError(old.ty(), new.ty()));
        }

        swap(old, &mut new);

        Ok(())
    }

    fn introduce(&mut self, variable_id: VariableId, initial: Constant) {
        //TODO: Can't happen, can it?
        assert_eq!(
            variable_id.0 + self.reference,
            self.values.len(),
            "variables on stack messed up"
        );

        //TODO: Typecheck?
        self.values.push(initial);
    }
}

impl Default for VariableValues {
    fn default() -> Self {
        Self::new()
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

    fn unwrap_on_outer_layer(self) -> Constant {
        self.value
    }

    fn some_or_please_return(&self) -> Option<&Constant> {
        if self.returning {
            None
        } else {
            Some(&self.value)
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

impl Rem for Constant {
    type Output = Result<Constant, RuntimeError>;

    fn rem(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Constant::Int(lhs), Constant::Int(rhs)) => Ok(Constant::Int((lhs % rhs)?)),
            (lhs, rhs) => Err(RuntimeError::InvalidBinaryOperation(
                lhs,
                BinaryOperator::Modulo,
                rhs,
            )),
        }
    }
}

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
        BinaryOperator::Modulo => lhs % rhs,
        //TODO error on ty mismatch
        BinaryOperator::Equals => Ok(Constant::Bool(lhs == rhs)),
        BinaryOperator::And => Ok(Constant::Bool(lhs.get_bool()? && rhs.get_bool()?)),
        BinaryOperator::Or => Ok(Constant::Bool(lhs.get_bool()? || rhs.get_bool()?)),
        BinaryOperator::Unequals => Ok(Constant::Bool(lhs != rhs)),
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
    Modulo,
    Equals,
    And,
    Or,
    Unequals,
}

impl Display for BinaryOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryOperator::Plus => write!(f, "+"),
            BinaryOperator::Times => write!(f, "*"),
            BinaryOperator::Minus => write!(f, "-"),
            BinaryOperator::Smaller => write!(f, "<"),
            BinaryOperator::Modulo => write!(f, "%"),
            BinaryOperator::Equals => write!(f, "=="),
            BinaryOperator::And => write!(f, "&&"),
            BinaryOperator::Or => write!(f, "||"),
            BinaryOperator::Unequals => write!(f, "!="),
        }
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum Stickyness {
    Disjunction,
    Conjunction,
    Comparison,
    Addition,
    Multiplication,
}

impl BinaryOperator {
    fn stickyness(&self) -> Stickyness {
        match self {
            BinaryOperator::Plus => Stickyness::Addition,
            BinaryOperator::Times => Stickyness::Multiplication,
            BinaryOperator::Minus => Stickyness::Addition,
            BinaryOperator::Smaller => Stickyness::Comparison,
            BinaryOperator::Modulo => Stickyness::Multiplication,
            BinaryOperator::Equals => Stickyness::Comparison,
            BinaryOperator::And => Stickyness::Conjunction,
            BinaryOperator::Or => Stickyness::Disjunction,
            BinaryOperator::Unequals => Stickyness::Comparison,
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

    fn len(&self) -> usize {
        self.variables.len()
    }

    fn cut_back_to(&mut self, old_stack_size: usize) {
        assert!(old_stack_size <= self.variables.len());
        self.variables.truncate(old_stack_size);
    }
}

impl From<Block> for Expr {
    fn from(block: Block) -> Self {
        Self::Block(block)
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

    fn get_fun_id(&mut self, word: Word) -> FunId {
        FunId(self.fun_names.insert_full(word).0)
    }

    fn parse_non_binary(&mut self) -> Result<Expr, ParseError> {
        //TODO: Combine with match, peek vs next issue
        if let Some(Token::BraceLeft) = self.peek() {
            return self.parse_block().map(Expr::Block);
        }

        let token = self.next()?;

        let mut expr = match token {
            Token::ParanLeft => {
                let inner_expr = self.parse_expr()?;
                self.expect(Token::ParanRight)?;
                inner_expr
            }
            Token::Negate => {
                let inner_expr = self.parse_non_binary()?;
                Expr::Negate(self.expressions.add(inner_expr))
            }
            Token::Ty(ty) => Expr::Constant(Constant::Ty(ty)),
            Token::Return => {
                let expr = self.parse_expr()?;

                Expr::Return(self.expressions.add(expr))
            }
            Token::Word(word) => match self.peek() {
                Some(Token::ParanLeft) => self.parse_fun_arguments(word, None)?,
                Some(Token::Assign) => {
                    self.expect(Token::Assign).expect("just peeked");
                    let expr = self.parse_expr()?;

                    Expr::Assign(self.stack.get(&word)?, self.expressions.add(expr))
                }
                _ =>
                //TODO: in doubt, should assume fun, not var.
                //TODO Err flow
                {
                    if let Some(internal_fun) = InternalFun::get(&word) {
                        Expr::Constant(Constant::Callable(Callable::InternalFun(internal_fun)))
                    } else if let Ok(variable_id) = self.stack.get(&word) {
                        Expr::Variable(variable_id)
                    } else {
                        Expr::Constant(Constant::Callable(Callable::Fun(self.get_fun_id(word))))
                    }
                }
            },

            Token::Literal(int_constant) => Expr::Constant(int_constant),
            _ => {
                return Err(ParseError::UnexpectedToken(
                    token,
                    "parse_non_binary".into(),
                ))
            }
        };

        while self.entertain(Token::Dot) {
            let name = self.expect_word()?;
            expr = self.parse_fun_arguments(name, Some(expr))?;
        }

        Ok(expr)
    }

    fn parse_follow_up(&mut self) -> Result<FollowUp, ParseError> {
        match self.peek() {
            Some(Token::BinaryOperator(op)) => {
                self.next().expect("just peeked");
                Ok(FollowUp::BinaryOperator(op))
            }
            None
            | Some(
                Token::Semicolon
                | Token::Comma
                | Token::BraceLeft
                | Token::ParanRight
                | Token::BraceRight,
            ) => Ok(FollowUp::End),
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

        let else_expr = if self.entertain(Token::Else) {
            match self.peek() {
                Some(Token::BraceLeft) => self.parse_block()?.into(),
                Some(Token::If) => self.parse_if()?,
                Some(unexpected) => {
                    return Err(ParseError::UnexpectedToken(
                        unexpected,
                        format!(
                            "expect '{}' or '{}' after '{}'",
                            Token::BraceLeft,
                            Token::If,
                            Token::Else,
                        ),
                    ))
                }
                None => return Err(ParseError::UnexpectedEndOfStream),
            }
        } else {
            Block {
                statements: ExprRange {
                    start: ExprId(0),
                    end: ExprId(0),
                },
            }
            .into()
        };

        let expr = Expr::If(
            self.expressions.add(cond),
            if_block,
            self.expressions.add(else_expr),
        );
        Ok(expr)
    }

    pub fn parse_while(&mut self) -> Result<Expr, ParseError> {
        self.expect(Token::While)?;

        let cond = self.parse_expr()?;

        //TODO: Check if boolean

        let while_block = self.parse_block()?;

        let expr = Expr::While(self.expressions.add(cond), while_block);
        Ok(expr)
    }

    pub fn parse_for(&mut self) -> Result<Expr, ParseError> {
        self.expect(Token::For)?;

        let introduced0 = self.expect_word()?;

        let introduced1 = if self.entertain(Token::Comma) {
            Some(self.expect_word()?)
        } else {
            None
        };

        self.expect(Token::In)?;

        let iterable = self.parse_expr()?;

        let old_stack_size = self.stack.len();
        let var0 = self.stack.push(introduced0)?;
        let var1 = if let Some(word) = introduced1 {
            Some(self.stack.push(word)?)
        } else {
            None
        };

        let block = self.parse_block()?;

        self.stack.cut_back_to(old_stack_size);

        let expr = Expr::For(var0, var1, self.expressions.add(iterable), block);
        Ok(expr)
    }

    pub fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        if let Some(Token::If) = self.peek() {
            return self.parse_if();
        }

        if let Some(Token::While) = self.peek() {
            return self.parse_while();
        }

        if let Some(Token::For) = self.peek() {
            return self.parse_for();
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
            let ty = self.expect_ty()?;
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
        let ty = self.expect_ty()?;

        for variable in variables.set.keys() {
            //TODO Test error
            self.stack.push(variable.clone())?;
        }

        let body = self.parse_block()?;

        self.stack.variables.clear();

        Ok(Fun::new(name, variables, ty, body))
    }

    fn expect_ty(&mut self) -> Result<Ty, ParseError> {
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
        allow_variable_introduction: bool,
        leading: Option<Expr>,
    ) -> Result<(Vec<Expr>, bool), ParseError> {
        self.expect(left_token)?;

        let mut statements = Vec::new();

        if let Some(leading) = leading {
            statements.push(leading);
        }

        let mut has_trailing_separator = false;

        loop {
            if self.entertain(right_token.clone()) {
                break;
            }

            if self.entertain(separator_token.clone()) {
                has_trailing_separator = true;
                continue;
            }

            if allow_variable_introduction && self.entertain(Token::Val) {
                let name = self.expect_word()?;

                //TODO Use type info
                let _ty = if self.entertain(Token::Colon) {
                    Some(self.expect_ty()?)
                } else {
                    None
                };

                self.expect(Token::Assign)?;

                //SYNC(INTRO_AFTER_EVAL) Variable gets introduced after evaluating its initial value.
                let expr = self.parse_expr()?;
                let variable_id = self.stack.push(name)?;

                statements.push(Expr::Introduce(variable_id, self.expressions.add(expr)));
                continue;
            }

            statements.push(self.parse_expr()?);
            has_trailing_separator = false;
        }

        Ok((statements, has_trailing_separator))
    }

    fn parse_block(&mut self) -> Result<Block, ParseError> {
        let old_stack_size = self.stack.len();
        let (mut statements, has_trailing_separator) = self
            .parse_expression_list_and_has_trailing_separator(
                Token::BraceLeft,
                Token::Semicolon,
                Token::BraceRight,
                true,
                None,
            )?;

        self.stack.cut_back_to(old_stack_size);

        if has_trailing_separator {
            let expr = Expr::Constant(Constant::None);
            statements.push(expr);
        }

        let range: ExprRange = self.expressions.add_range(statements);

        Ok(Block::new(range))
    }

    fn parse_fun_arguments(
        &mut self,
        name: Word,
        leading: Option<Expr>,
    ) -> Result<Expr, ParseError> {
        let (statements, _) = self.parse_expression_list_and_has_trailing_separator(
            Token::ParanLeft,
            Token::Comma,
            Token::ParanRight,
            false,
            leading,
        )?;

        let range = self.expressions.add_range(statements);

        let callable = if let Some(internal_fun) = InternalFun::get(&name) {
            Callable::InternalFun(internal_fun)
        } else {
            Callable::Fun(self.get_fun_id(name))
        };

        Ok(Expr::Call(Call {
            callable,
            arguments: range,
        }))
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
                        format!(
                            "expected a top level declaration (currently only '{}')",
                            Token::Fun
                        ),
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
    fn evaluate_block_unscoped(
        &self,
        block: &Block,
        variable_values: &mut VariableValues,
    ) -> Result<Evaluation, RuntimeError> {
        if block.statements.start == block.statements.end {
            return Ok(Constant::None.into());
        }
        assert!(block.statements.start < block.statements.end);

        let last_id = block.statements.end.0.checked_sub(1).expect("assert above");

        for id in block.statements.start.0..last_id {
            let expr_id = ExprId(id);
            let eval = self.evaluate_expr(expr_id, variable_values)?;
            let Some(_) = eval.some_or_please_return() else {
                return Ok(eval);
            };
        }

        self.evaluate_expr(ExprId(last_id), variable_values)
    }

    fn evaluate_block(
        &self,
        block: &Block,
        variable_values: &mut VariableValues,
    ) -> Result<Evaluation, RuntimeError> {
        let old_size = variable_values.values.len();
        let old_reference = variable_values.reference;

        let res = self.evaluate_block_unscoped(block, variable_values)?;

        assert!(old_size <= variable_values.values.len());
        assert_eq!(variable_values.reference, old_reference);
        variable_values.values.truncate(old_size);

        Ok(res)
    }

    fn evaluate_expr(
        &self,
        expr_id: ExprId,
        variable_values: &mut VariableValues,
    ) -> Result<Evaluation, RuntimeError> {
        match self.expressions.get(expr_id) {
            Expr::Constant(int_constant) => Ok((int_constant.clone()).into()),
            Expr::Variable(variable_id) => Ok(variable_values.get(variable_id)?.into()),
            Expr::BinaryOp(lhs_expr, op, rhs_expr) => {
                let lhs = self.evaluate_expr(*lhs_expr, variable_values)?;
                let Some(lhs) = lhs.some_or_please_return() else {
                    return Ok(lhs);
                };

                // Short circuiting
                if op == &BinaryOperator::And && !lhs.get_bool()? {
                    return Ok(Constant::Bool(false).into());
                }

                if op == &BinaryOperator::Or && lhs.get_bool()? {
                    return Ok(Constant::Bool(true).into());
                }

                let rhs = self.evaluate_expr(*rhs_expr, variable_values)?;
                let Some(rhs) = rhs.some_or_please_return() else {
                    return Ok(rhs);
                };

                Ok(combine_with_operator(lhs.clone(), *op, rhs.clone())?.into())
            }
            Expr::Block(block) => self.evaluate_block(block, variable_values),
            Expr::Return(expr) => self
                .evaluate_expr(*expr, variable_values)
                .map(|evaluation| evaluation.returning()),
            Expr::If(expr, if_block, else_expr) => {
                let eval = self.evaluate_expr(*expr, variable_values)?;
                let Some(constant) = eval.some_or_please_return() else {
                    return Ok(eval);
                };

                let b = constant.get_bool()?;

                if b {
                    self.evaluate_block(if_block, variable_values)
                } else {
                    self.evaluate_expr(*else_expr, variable_values)
                }
            }
            Expr::Call(call) => {
                let mut parameters = Vec::new();
                for id in call.arguments.start.0..call.arguments.end.0 {
                    let eval = self.evaluate_expr(ExprId(id), variable_values)?;
                    let Some(parameter) = eval.some_or_please_return() else {
                        return Ok(eval);
                    };

                    parameters.push(parameter.clone());
                }

                self.evaluate_call(call.callable, parameters, variable_values)
                    .map(|c| c.into())
            }
            Expr::Assign(variable_id, expr_id) => {
                let eval = self.evaluate_expr(*expr_id, variable_values)?;
                let Some(eval) = eval.some_or_please_return() else {
                    return Ok(eval);
                };
                variable_values.set(*variable_id, eval.clone())?;
                Ok(Constant::None.into())
            }
            Expr::Introduce(variable_id, expr_id) => {
                //SYNC(INTRO_AFTER_EVAL) Variable gets introduced after evaluating its initial value.
                let eval = self.evaluate_expr(*expr_id, variable_values)?;
                let Some(eval) = eval.some_or_please_return() else {
                    return Ok(eval);
                };
                variable_values.introduce(*variable_id, eval.clone());

                Ok(Constant::None.into())
            }
            Expr::While(expr_id, block) => loop {
                let eval = self.evaluate_expr(*expr_id, variable_values)?;
                let Some(constant) = eval.some_or_please_return() else {
                    return Ok(eval);
                };

                let b = constant.get_bool()?;

                if !b {
                    return Ok(Constant::None.into());
                }

                let block_result = self.evaluate_block(block, variable_values)?;
                let Some(_) = block_result.some_or_please_return() else {
                    return Ok(block_result);
                };
            },
            Expr::For(var0, var1, iterable, block) => {
                let iterable = self.evaluate_expr(*iterable, variable_values)?;
                let Some(iterable) = iterable.some_or_please_return() else {
                    return Ok(iterable);
                };

                let iter = match iterable {
                    Constant::List(rc) => rc.borrow().clone(),
                    _ => return Err(RuntimeError::NotIterable(iterable.clone())),
                };

                match var1 {
                    Some(var1) => {
                        for (i, c) in iter.into_iter().enumerate() {
                            variable_values
                                .introduce(*var0, Constant::Int(IntConstant::Small(i as i128)));
                            variable_values.introduce(*var1, c);

                            let eval = self.evaluate_block(block, variable_values)?;

                            variable_values.values.pop();
                            variable_values.values.pop();

                            let Some(_) = eval.some_or_please_return() else {
                                return Ok(eval);
                            };
                        }
                    }
                    None => {
                        for c in iter.into_iter() {
                            variable_values.introduce(*var0, c);

                            let eval = self.evaluate_block(block, variable_values)?;

                            variable_values.values.pop();

                            let Some(_) = eval.some_or_please_return() else {
                                return Ok(eval);
                            };
                        }
                    }
                }

                Ok(Constant::None.into())
            }
            Expr::Negate(expr_id) => {
                let inner_value = self.evaluate_expr(*expr_id, variable_values)?;
                let Some(inner_value) = inner_value.some_or_please_return() else {
                    return Ok(inner_value);
                };
                Ok(Constant::Bool(!inner_value.get_bool()?).into())
            }
        }
    }

    fn evaluate_call(
        &self,
        callable: Callable,
        parameters: Vec<Constant>,
        variable_values: &mut VariableValues,
    ) -> Result<Constant, RuntimeError> {
        let result = match callable {
            Callable::Fun(fun_id) => self.evaluate_fun(fun_id, parameters, variable_values),
            Callable::InternalFun(internal_fun) => {
                self.evaluate_internal_fun(internal_fun, parameters, variable_values)
            }
        }?;

        Ok(result)
    }

    pub fn evaluate_fun(
        &self,
        fun_id: FunId,
        //TODO: This allocation slows us down.
        parameters: Vec<Constant>,
        variable_values: &mut VariableValues,
    ) -> Result<Constant, RuntimeError> {
        let fun = self
        .funs
        .get(fun_id.0)
        .unwrap() //TODO
        // .ok_or_else(|| RuntimeError::UnknownFunction(name.clone()))?
        ;

        debug!(
            "calling {} ({} parameters) with stack size {} and old reference {}",
            fun.name,
            fun.variables.len(),
            variable_values.values.len(),
            variable_values.reference
        );

        if fun.variables.len() != parameters.len() {
            return Err(RuntimeError::IncompatibleParameters(
                parameters.iter().map(|c| c.ty()).collect(),
                fun.variables.clone(),
                fun.name.clone(),
            ));
        }

        let old_reference = variable_values.reference;
        variable_values.reference = variable_values.values.len();

        for (id, parameter) in parameters.into_iter().enumerate() {
            variable_values.introduce(VariableId(id), parameter);
        }

        let result = self
            .evaluate_block(&fun.body, variable_values)?
            .unwrap_on_outer_layer();

        assert_eq!(
            variable_values.values.len() - variable_values.reference,
            fun.variables.len()
        );

        //TODO: Check when parsing fun
        if result.ty() != fun.ty {
            return Err(RuntimeError::WrongReturnType(
                fun.name.clone(),
                fun.ty,
                result.ty(),
            ));
        }

        variable_values.values.truncate(variable_values.reference);
        variable_values.reference = old_reference;

        Ok(result)
    }

    pub fn evaluate_internal_fun(
        &self,
        internal_fun: InternalFun,
        parameters: Vec<Constant>,
        variable_values: &mut VariableValues,
    ) -> Result<Constant, RuntimeError> {
        match internal_fun {
            InternalFun::New => {
                if let Some(&Constant::Ty(ty)) = parameters.iter().collect_single() {
                    match ty {
                        Ty::List => return Ok(Constant::List(Default::default())),
                        _ => return Err(RuntimeError::StaticFunctionOnWrongType(ty, internal_fun)),
                    }
                }
            }
            InternalFun::Push => {
                if let Some((Constant::List(list), c)) = parameters.iter().collect_tuple() {
                    (*list).borrow_mut().push(c.clone());
                    return Ok(Constant::None);
                }
            }
            InternalFun::Get => {
                if let Some((Constant::List(list), &Constant::Int(i))) =
                    parameters.iter().collect_tuple()
                {
                    let IntConstant::Small(id) = i;
                    let id: usize = id
                        .try_into()
                        .map_err(|_| RuntimeError::IndexOutOfBounds(list.borrow().clone(), i))?;
                    return list
                        .borrow()
                        .get(id)
                        .ok_or_else(|| RuntimeError::IndexOutOfBounds(list.borrow().clone(), i))
                        .cloned();
                }
            }

            InternalFun::Pop => {
                if let Some(Constant::List(list)) = parameters.iter().collect_single() {
                    return list.borrow_mut().pop().ok_or_else(|| {
                        RuntimeError::InternalOperationInvalid(internal_fun, parameters.clone())
                    });
                }
            }
            InternalFun::Len => {
                if let Some(Constant::List(list)) = parameters.iter().collect_single() {
                    return Ok(Constant::Int(IntConstant::Small(
                        //TODO Cast so ok?
                        list.borrow().len() as i128,
                    )));
                }
            }
            InternalFun::FromFile => {
                if let Some((&Constant::Ty(ty), Constant::List(list))) =
                    parameters.iter().collect_tuple()
                {
                    match ty {
                        Ty::List => {
                            let path = try_list_to_string(&list.borrow())?;

                            let file_content =
                                read_to_string(&path).map_err(|err| RuntimeError::Io(err, path))?;
                            let list = file_content.chars().map(Constant::Char).collect_vec();
                            return Ok(list.into());
                        }
                        _ => return Err(RuntimeError::StaticFunctionOnWrongType(ty, internal_fun)),
                    }
                }
            }
            InternalFun::SplitWhitespace => {
                if let Some(Constant::List(list)) = parameters.iter().collect_single() {
                    let string = try_list_to_string(&list.borrow())?;
                    let result: Vec<Constant> = string
                        .split_whitespace()
                        .map(|sub| str_to_list(sub).into())
                        .collect_vec();

                    return Ok(result.into());
                }
            }
            InternalFun::Parse => {
                if let Some((Constant::List(list), Constant::Ty(ty))) =
                    parameters.iter().collect_tuple()
                {
                    let string = try_list_to_string(&list.borrow())?;

                    let res = match ty {
                        Ty::Int => {
                            let i: i128 = string
                                .parse()
                                .map_err(|err| RuntimeError::Parse(string, Ty::Int, err))?;
                            Constant::Int(IntConstant::Small(i))
                        }
                        _ => {
                            return Err(RuntimeError::InternalOperationInvalid(
                                internal_fun,
                                parameters,
                            ))
                        }
                    };

                    return Ok(res);
                }
            }
            InternalFun::Sort => {
                if let Some(Constant::List(list)) = parameters.iter().collect_single() {
                    let mut new_list = Vec::new();
                    for val in list.borrow().iter() {
                        let &Constant::Int(i) = val else {
                            return Err(RuntimeError::TypeError(Ty::Int, val.ty()));
                        };
                        new_list.push(i);
                    }
                    new_list.sort();
                    let new_list = new_list.into_iter().map(Constant::Int).collect();

                    *list.borrow_mut() = new_list;

                    return Ok(Constant::None);
                }
            }
            InternalFun::Abs => {
                if let Some(Constant::Int(i)) = parameters.iter().collect_single() {
                    let IntConstant::Small(i) = i;
                    return Ok(Constant::Int(IntConstant::Small(i.abs())));
                }
            }
            InternalFun::Lines => {
                if let Some(Constant::List(list)) = parameters.iter().collect_single() {
                    let string = try_list_to_string(&list.borrow())?;
                    let result: Vec<Constant> = string
                        .lines()
                        .map(|sub| str_to_list(sub).into())
                        .collect_vec();

                    return Ok(result.into());
                }
            }
            InternalFun::Debug => {
                if let Some(c) = parameters.iter().collect_single() {
                    info!("[Debug] {c}");
                    return Ok(c.clone());
                }
            }
            InternalFun::Invoke => {
                let mut params_iter = parameters.iter();
                if let Some(&Constant::Callable(callable)) = params_iter.next() {
                    //TODO Collect meh
                    return self.evaluate_call(
                        callable,
                        params_iter.cloned().collect(),
                        variable_values,
                    );
                }
            }
        }

        Err(RuntimeError::IncompatibleParametersForInternal(
            parameters.iter().map(|c| c.ty()).collect(),
            internal_fun,
        ))
    }

    pub fn evaluate_main(&self) -> Result<Constant, RuntimeError> {
        let mut variable_values = VariableValues::new();

        self.evaluate_fun(
            self.entry_point.ok_or(RuntimeError::NoEntryPoint)?,
            Vec::new(),
            &mut variable_values,
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

fn try_list_to_string(list: &Vec<Constant>) -> Result<String, RuntimeError> {
    let mut s = String::new();

    for c in list {
        let Constant::Char(c) = c else {
            //TODO test case for that; maybe better error message due to context
            return Err(RuntimeError::TypeError(Ty::Char, c.ty()));
        };

        s.push(*c);
    }

    Ok(s)
}

fn str_to_list(s: &str) -> Vec<Constant> {
    s.chars().map(Constant::Char).collect()
}

#[derive(Debug)]
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
    #[error("unknown character: '{0}'")]
    UnknownCharacter(char),
    #[error("invalid word: '{0}'")]
    InvalidWord(String),
    #[error("missing '\"' to end the string literal")]
    MissingEndOfStringLiteral,
    #[error("char literal invalid")]
    InvalidCharLiteral,
}

struct Parsee {
    //Should not be empty.
    lines: Vec<Vec<char>>,
    current_line_id: usize,
    current_char_id: usize,
    filename: String,
}

impl Parsee {
    fn skip_if_end_of_line(&mut self) {
        while self
            .lines
            .get(self.current_line_id)
            .is_some_and(|line| line.len() == self.current_char_id)
        {
            self.current_line_id += 1;
            self.current_char_id = 0;
        }
    }

    fn peek(&self) -> Option<char> {
        self.lines
            .get(self.current_line_id)
            .map(|line| line[self.current_char_id])
    }

    fn next(&mut self) -> Option<char> {
        let c = self.peek()?;

        self.current_char_id += 1;
        self.skip_if_end_of_line();

        Some(c)
    }

    fn new(chars: &str, filename: String) -> Self {
        let lines = chars.lines().map(|line| line.chars().collect()).collect();
        let mut res = Self {
            lines,
            current_line_id: 0,
            current_char_id: 0,
            filename,
        };
        res.skip_if_end_of_line();
        res
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
            if !(next.is_ascii_alphanumeric() || next == '_') {
                break;
            }

            word.push(next);
            self.next();
        }

        if let Some(token) = keyword_to_token(&word) {
            return Ok(token.clone());
        }

        Ok(Token::Word(
            word.try_into().expect("just parsed it correctly"),
        ))
    }

    fn parse_token(&mut self) -> Result<Option<Token>, SourceTokenizeError> {
        self.skip_whitespaces();

        let Some(start) = self.next() else {
            return Ok(None);
        };

        if start == '/' && self.peek() == Some('/') {
            self.skip_rest_of_line();
            return self.parse_token();
        }

        if let Some(token) = single_digit_to_token(start) {
            if matches!(token, Token::BinaryOperator(BinaryOperator::Minus))
                && Some('>') == self.peek()
            {
                self.next();
                return Ok(Some(Token::Arrow));
            }

            if matches!(token, Token::Assign) && Some('=') == self.peek() {
                self.next();
                return Ok(Some(Token::BinaryOperator(BinaryOperator::Equals)));
            }

            if matches!(token, Token::BitAnd) && Some('&') == self.peek() {
                self.next();
                return Ok(Some(Token::BinaryOperator(BinaryOperator::And)));
            }

            if matches!(token, Token::BitOr) && Some('|') == self.peek() {
                self.next();
                return Ok(Some(Token::BinaryOperator(BinaryOperator::Or)));
            }

            if matches!(token, Token::Negate) && Some('=') == self.peek() {
                self.next();
                return Ok(Some(Token::BinaryOperator(BinaryOperator::Unequals)));
            }

            return Ok(Some(token.clone()));
        }

        let token = match start {
            '0'..='9' => self.parse_number(start)?,
            'A'..='z' => self.parse_word(start)?,
            '"' => self.parse_string_literal()?,
            '\'' => self.parse_char()?,
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

        Ok(Token::Literal(Constant::Int(IntConstant::Small(i))))
    }

    fn error(&self, error: TokenizeError) -> SourceTokenizeError {
        SourceTokenizeError::new(
            self.filename.clone(),
            self.current_line_id,
            self.current_char_id,
            self.current_line_to_string(),
            error,
        )
    }

    fn current_line_to_string(&self) -> String {
        self.lines
            .get(self.current_line_id)
            .expect("hopefully only called when still pointing to something")
            .iter()
            .collect()
    }

    fn skip_rest_of_line(&mut self) {
        self.current_char_id = self.lines[self.current_line_id].len();
        self.skip_if_end_of_line();
    }

    fn parse_char(&mut self) -> Result<Token, SourceTokenizeError> {
        //TODO test error
        let Some(c) = self.next() else {
            return Err(self.error(TokenizeError::InvalidCharLiteral));
        };
        let Some('\'') = self.next() else {
            return Err(self.error(TokenizeError::InvalidCharLiteral));
        };

        Ok(Token::Literal(Constant::Char(c)))
    }

    fn parse_string_literal(&mut self) -> Result<Token, SourceTokenizeError> {
        let mut list = Vec::new();

        loop {
            //TODO test error
            let Some(c) = self.next() else {
                return Err(self.error(TokenizeError::MissingEndOfStringLiteral));
            };
            if c == '"' {
                return Ok(Token::Literal(list.into()));
            }

            list.push(Constant::Char(c));
        }
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

fn single_digit_to_token(c: char) -> Option<Token> {
    match c {
        ',' => Some(Token::Comma),
        ';' => Some(Token::Semicolon),
        '+' => Some(Token::BinaryOperator(BinaryOperator::Plus)),
        '-' => Some(Token::BinaryOperator(BinaryOperator::Minus)),
        '*' => Some(Token::BinaryOperator(BinaryOperator::Times)),
        '<' => Some(Token::BinaryOperator(BinaryOperator::Smaller)),
        '%' => Some(Token::BinaryOperator(BinaryOperator::Modulo)),
        '(' => Some(Token::ParanLeft),
        ')' => Some(Token::ParanRight),
        '{' => Some(Token::BraceLeft),
        '}' => Some(Token::BraceRight),
        ':' => Some(Token::Colon),
        '=' => Some(Token::Assign),
        '.' => Some(Token::Dot),
        '&' => Some(Token::BitAnd),
        '|' => Some(Token::BitOr),
        '!' => Some(Token::Negate),
        _ => None,
    }
}

fn keyword_to_token(keyword: &str) -> Option<Token> {
    if keyword == "fun" {
        Some(Token::Fun)
    } else if keyword == "return" {
        Some(Token::Return)
    } else if keyword == "List" {
        Some(Token::Ty(Ty::List))
    } else if keyword == "Range" {
        Some(Token::Ty(Ty::Range))
    } else if keyword == "int" {
        Some(Token::Ty(Ty::Int))
    } else if keyword == "None" {
        Some(Token::Ty(Ty::None))
    } else if keyword == "if" {
        Some(Token::If)
    } else if keyword == "else" {
        Some(Token::Else)
    } else if keyword == "while" {
        Some(Token::While)
    } else if keyword == "for" {
        Some(Token::For)
    } else if keyword == "in" {
        Some(Token::In)
    } else if keyword == "call" {
        Some(Token::Call)
    } else if keyword == "val" {
        Some(Token::Val)
    } else if keyword == "bool" {
        Some(Token::Ty(Ty::Bool))
    } else if keyword == "char" {
        Some(Token::Ty(Ty::Char))
    } else if keyword == "Callable" {
        Some(Token::Ty(Ty::Callable))
    } else if keyword == "true" {
        Some(Token::Literal(Constant::Bool(true)))
    } else if keyword == "false" {
        Some(Token::Literal(Constant::Bool(false)))
    } else {
        None
    }
}

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

            if !chars.all(|c| c.is_ascii_alphanumeric() || c == '_') {
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
    List,
    Range,
    Ty,
    Callable,
    Char,
}

impl Display for Ty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Ty::List => write!(f, "List"),
            Ty::Int => write!(f, "int"),
            Ty::Bool => write!(f, "bool"),
            Ty::Range => write!(f, "Range"),
            Ty::None => write!(f, "None"),
            Ty::Ty => write!(f, "TYPE"),
            Ty::Char => write!(f, "char"),
            Ty::Callable => write!(f, "Callable"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Token {
    Return,
    Else,
    Ty(Ty),
    Word(Word),
    Literal(Constant),
    BinaryOperator(BinaryOperator),
    Semicolon,
    Colon,
    BitAnd,
    BitOr,
    Negate,
    While,
    Comma,
    Assign,
    Arrow,
    ParanLeft,
    ParanRight,
    BraceLeft,
    BraceRight,
    Fun,
    Val,
    If,
    Call,
    Dot,
    For,
    In,
}

impl Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::Return => write!(f, "return"),
            Token::Ty(ty) => write!(f, "{ty}"),
            Token::Word(word) => write!(f, "{word}"),
            Token::Literal(int_constant) => write!(f, "{int_constant}"),
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
            Token::Else => write!(f, "else"),
            Token::Call => write!(f, "call"),
            Token::Assign => write!(f, "="),
            Token::Dot => write!(f, "."),
            Token::Val => write!(f, "val"),
            Token::While => write!(f, "while"),
            Token::For => write!(f, "for"),
            Token::In => write!(f, "in"),
            Token::BitAnd => write!(f, "&"),
            Token::BitOr => write!(f, "|"),
            Token::Negate => write!(f, "!"),
        }
    }
}

pub trait MyItertools: Iterator {
    fn collect_single(mut self) -> Option<Self::Item>
    where
        Self: Sized,
    {
        let item = self.next()?;

        if self.next().is_some() {
            None
        } else {
            Some(item)
        }
    }
}

impl<T> MyItertools for T where T: Iterator + ?Sized {}
