use indexmap::{indexmap, indexset, IndexMap, IndexSet};
use itertools::Itertools;
use log::{debug, info};
use std::hash::Hash;
use std::iter::once;
use std::num::TryFromIntError;
use std::ops::{Div, Neg};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;
use std::time::Instant;
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
    //TODO very hard to read variables debug print
    IncompatibleParameters(Vec<Ty>, Variables, Word),
    #[error("arguments '{0:?}' incompatible with internal function {1}")]
    IncompatibleParametersForInternal(Vec<Ty>, InternalFun),
    #[error("unknown function '{0}'")]
    UnknownDotFunction(Word),
    //TODO constuctor, make sure they are unequal
    #[error("type error: expected '{0}', found '{1}'")]
    TypeError(Ty, Ty),
    #[error("invalid operation:'{0} {1} {2}'")]
    InvalidBinaryOperation(Constant, BinaryOperator, Constant),
    #[error("function '{0}' has return type '{1}', but evaluated to '{2}'")]
    WrongReturnType(Word, GeneralTy, Ty),
    #[error("missing entry point (function '{}')", *MAIN)]
    NoEntryPoint,
    #[error("index '{1}' out of bounds for list {0})")]
    IndexOutOfBounds(Constant, IntConstant),
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
    #[error("key '{0}' already exists")]
    KeyAlreadyExists(HashableConstant),
    #[error("key '{0}' does not exist")]
    KeyDoesNotExist(HashableConstant),
    //TODO Test if we now can call variables without evoke.
    #[error("cannot call '{0}'")]
    NotCallable(Constant),
    #[error("cannot *exactly* divide '{0}' by '{1}'")]
    DivisionNotExact(IntConstant, IntConstant),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IntConstant {
    Small(i128),
}

impl From<IntConstant> for Constant {
    fn from(value: IntConstant) -> Self {
        Constant::HashableConstant(HashableConstant::PrimitiveConstant(PrimitiveConstant::Int(
            value,
        )))
    }
}

impl Neg for IntConstant {
    type Output = IntConstant;

    fn neg(self) -> Self::Output {
        match self {
            IntConstant::Small(i) => IntConstant::Small(-i),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PrimitiveConstant {
    Int(IntConstant),
    Bool(bool),
    None,
    Ty(Ty),
    Char(char),
    Callable(Callable),
}

impl TryInto<usize> for IntConstant {
    type Error = TryFromIntError;

    fn try_into(self) -> Result<usize, Self::Error> {
        let IntConstant::Small(id) = self;
        id.try_into()
    }
}

impl PrimitiveConstant {
    fn ty(&self) -> Ty {
        match self {
            Self::Int(_) => Ty::Int,
            Self::Bool(_) => Ty::Bool,
            Self::None => Ty::None,
            Self::Ty(_) => Ty::Ty,
            Self::Char(_) => Ty::Char,
            Self::Callable(_) => Ty::Callable,
        }
    }

    fn deep_clone(&self) -> PrimitiveConstant {
        self.clone()
    }
}

impl From<PrimitiveConstant> for Constant {
    fn from(value: PrimitiveConstant) -> Self {
        Constant::HashableConstant(HashableConstant::PrimitiveConstant(value))
    }
}

impl HashableConstant {
    fn ty(&self) -> Ty {
        match self {
            HashableConstant::PrimitiveConstant(primitive_constant) => primitive_constant.ty(),
            HashableConstant::List(_) => Ty::List,
        }
    }

    fn deep_clone(&self) -> HashableConstant {
        match self {
            HashableConstant::PrimitiveConstant(primitive_constant) => {
                HashableConstant::PrimitiveConstant(primitive_constant.deep_clone())
            }
            HashableConstant::List(rc) => {
                let clone = rc
                    .borrow()
                    .iter()
                    .map(HashableConstant::deep_clone)
                    .collect();

                HashableConstant::List(Rc::new(RefCell::new(clone)))
            }
        }
    }
}

impl PartialEq for HashableConstant {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (HashableConstant::PrimitiveConstant(s), HashableConstant::PrimitiveConstant(o)) => {
                s == o
            }
            (HashableConstant::List(s), HashableConstant::List(o)) => *s.borrow() == *o.borrow(),
            (s, o) => {
                assert_ne!(s.ty(), o.ty(), "implementation missing for {}", s.ty());
                false
            }
        }
    }
}

impl Eq for HashableConstant {}

impl Hash for HashableConstant {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            HashableConstant::PrimitiveConstant(primitive_constant) => {
                state.write_u8(0);
                primitive_constant.hash(state);
            }
            HashableConstant::List(rc) => {
                state.write_u8(1);
                (*rc.borrow()).hash(state);
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum HashableConstant {
    PrimitiveConstant(PrimitiveConstant),
    List(Rc<RefCell<Vec<HashableConstant>>>),
}

#[derive(Clone, Debug)]
pub enum Constant {
    HashableConstant(HashableConstant),
    Dict(Rc<RefCell<IndexMap<HashableConstant, Constant>>>),
}

impl Default for Constant {
    fn default() -> Self {
        Self::HashableConstant(HashableConstant::PrimitiveConstant(PrimitiveConstant::None))
    }
}

impl PartialEq for Constant {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::HashableConstant(s), Self::HashableConstant(o)) => s == o,
            (Self::Dict(s), Self::Dict(o)) => *s.borrow() == *o.borrow(),
            (s, o) => {
                assert_ne!(s.ty(), o.ty(), "implementation missing for {}", s.ty());
                false
            }
        }
    }
}

impl Eq for Constant {}

impl Constant {
    fn get_bool(&self) -> Result<bool, RuntimeError> {
        if let Constant::HashableConstant(HashableConstant::PrimitiveConstant(
            PrimitiveConstant::Bool(b),
        )) = self
        {
            Ok(*b)
        } else {
            let actual = self.ty();
            let expected = Ty::Bool;
            assert_ne!(actual, expected);
            Err(RuntimeError::TypeError(expected, actual))
        }
    }

    fn get_int(&self) -> Result<IntConstant, RuntimeError> {
        if let Constant::HashableConstant(HashableConstant::PrimitiveConstant(
            PrimitiveConstant::Int(i),
        )) = self
        {
            Ok(*i)
        } else {
            let actual = self.ty();
            let expected = Ty::Int;
            assert_ne!(actual, expected);
            Err(RuntimeError::TypeError(expected, actual))
        }
    }

    fn ty(&self) -> Ty {
        match self {
            Constant::Dict(_) => Ty::Dict,
            Constant::HashableConstant(hashable_constant) => hashable_constant.ty(),
        }
    }

    fn deep_clone(&self) -> Self {
        match self {
            Constant::HashableConstant(hashable_constant) => {
                Constant::HashableConstant(hashable_constant.deep_clone())
            }
            Constant::Dict(rc) => {
                let clone = rc
                    .borrow()
                    .iter()
                    .map(|(k, v)| (k.deep_clone(), v.deep_clone()))
                    .collect();

                Constant::Dict(Rc::new(RefCell::new(clone)))
            }
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

impl Display for PrimitiveConstant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Int(int) => write!(f, "{int}"),
            Self::Bool(b) => write!(f, "{b}"),
            Self::None => write!(f, "()"),
            //TODO make it good
            Self::Ty(ty) => write!(f, "{ty}"),
            Self::Char(c) => write!(f, "{c}"),
            Self::Callable(callable) => write!(f, "{callable}"),
        }
    }
}

impl Display for HashableConstant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PrimitiveConstant(primitive_constant) => write!(f, "{primitive_constant}"),
            //TODO make it good
            Self::List(vec) => {
                let vec = &*vec.borrow();
                if !vec.is_empty() && vec.iter().all(|hc| hc.ty() == Ty::Char) {
                    write!(f, "\"")?;
                    for elem in vec {
                        write!(f, "{elem}")?;
                    }
                    write!(f, "\"")
                } else {
                    write!(f, "[")?;
                    for elem in vec {
                        write!(f, "{elem}, ")?; //TODO trailing comma
                    }
                    write!(f, "]")
                }
            }
        }
    }
}

impl Display for Constant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HashableConstant(hashable_constant) => write!(f, "{hashable_constant}"),
            //TODO make it good
            Self::Dict(index_map) => {
                write!(f, "{{")?;
                for (k, v) in &*index_map.borrow() {
                    write!(f, "{k} -> {v}, ")?; //TODO trailing comma
                }
                write!(f, "}}")
            }
        }
    }
}

impl Rem for IntConstant {
    type Output = Result<IntConstant, RuntimeError>;

    fn rem(self, other: Self) -> Self::Output {
        match (self, other) {
            (IntConstant::Small(lhs), IntConstant::Small(rhs)) => lhs
                .checked_rem_euclid(rhs)
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

impl Div for IntConstant {
    type Output = Result<IntConstant, RuntimeError>;

    fn div(self, other: Self) -> Self::Output {
        if (self % other)? != Self::Small(0) {
            return Err(RuntimeError::DivisionNotExact(self, other));
        }

        match (self, other) {
            (IntConstant::Small(lhs), IntConstant::Small(rhs)) => lhs
                .checked_div(rhs)
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
    Minus(ExprId),
}

#[derive(Clone, Copy, Debug)]
pub struct VariableId(usize);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct ExprId(usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ListId(usize);

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
//TODO pub field
pub struct FunId {
    module_id: ModuleId,
    fun_in_module: usize,
}

impl FunId {
    fn new(module_id: ModuleId, fun_in_module: usize) -> Self {
        Self {
            module_id,
            fun_in_module,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
//TODO pub field
pub struct ModuleId(pub usize);

impl Display for ModuleId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Display for FunId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}->{}", self.module_id, self.fun_in_module)
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
    callable: ExprId,
    arguments: ExprRange,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

#[derive(Clone, Copy, Debug, PartialEq, Hash, Eq)]
pub enum InternalFun {
    New,
    Remove,
    Invoke,
    Debug,
    Push,
    Get,
    Set,
    Pop,
    FromFile,
    Len,
    SplitWhitespace,
    Parse,
    Sort,
    Abs,
    Lines,
    SetNew,
    Has,
    Keys,
    DeepClone,
    UpdateExisting,
    Intersection,
    Xor,
}

impl Display for InternalFun {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InternalFun::New => write!(f, "new"),
            InternalFun::Push => write!(f, "push"),
            InternalFun::Get => write!(f, "get"),
            InternalFun::Set => write!(f, "set"),
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
            InternalFun::SetNew => write!(f, "set_new"),
            InternalFun::Has => write!(f, "has"),
            InternalFun::Keys => write!(f, "keys"),
            InternalFun::DeepClone => write!(f, "deep_clone"),
            InternalFun::Remove => write!(f, "remove"),
            InternalFun::UpdateExisting => write!(f, "update_existing"),
            InternalFun::Xor => write!(f, "xor"),
            InternalFun::Intersection => write!(f, "intersection"),
        }
    }
}

static WORD_INTERNAL_FUN_MAP: LazyLock<IndexMap<Word, InternalFun>> = LazyLock::new(|| {
    indexmap! {
        "new".try_into().expect("const") => InternalFun::New,
        "push".try_into().expect("const") => InternalFun::Push,
        "get".try_into().expect("const") => InternalFun::Get,
        "set".try_into().expect("const") => InternalFun::Set,
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
        "set_new".try_into().expect("const") => InternalFun::SetNew,
        "has".try_into().expect("const") => InternalFun::Has,
        "deep_clone".try_into().expect("const") => InternalFun::DeepClone,
        "keys".try_into().expect("const") => InternalFun::Keys,
        "remove".try_into().expect("const") => InternalFun::Remove,
        "update_existing".try_into().expect("const") => InternalFun::UpdateExisting,
        "xor".try_into().expect("const") => InternalFun::Xor,
        "intersection".try_into().expect("const") => InternalFun::Intersection,
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
    map: IndexMap<Word, Variable>,
}
impl Variables {
    fn new() -> Self {
        Self {
            map: IndexMap::new(),
        }
    }

    fn insert_new(&mut self, variable: Variable) -> Result<(), ParseError> {
        match self.map.entry(variable.name.clone()) {
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
        self.map.len()
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
            (
                Constant::HashableConstant(HashableConstant::PrimitiveConstant(
                    PrimitiveConstant::Int(lhs),
                )),
                Constant::HashableConstant(HashableConstant::PrimitiveConstant(
                    PrimitiveConstant::Int(rhs),
                )),
            ) => Ok(Constant::HashableConstant(
                HashableConstant::PrimitiveConstant(PrimitiveConstant::Int((lhs + rhs)?)),
            )),
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
            (
                Constant::HashableConstant(HashableConstant::PrimitiveConstant(
                    PrimitiveConstant::Int(lhs),
                )),
                Constant::HashableConstant(HashableConstant::PrimitiveConstant(
                    PrimitiveConstant::Int(rhs),
                )),
            ) => Ok(Constant::HashableConstant(
                HashableConstant::PrimitiveConstant(PrimitiveConstant::Int((lhs * rhs)?)),
            )),
            (lhs, rhs) => Err(RuntimeError::InvalidBinaryOperation(
                lhs,
                BinaryOperator::Times,
                rhs,
            )),
        }
    }
}

impl Div for Constant {
    type Output = Result<Constant, RuntimeError>;

    fn div(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (
                Constant::HashableConstant(HashableConstant::PrimitiveConstant(
                    PrimitiveConstant::Int(lhs),
                )),
                Constant::HashableConstant(HashableConstant::PrimitiveConstant(
                    PrimitiveConstant::Int(rhs),
                )),
            ) => Ok(Constant::HashableConstant(
                HashableConstant::PrimitiveConstant(PrimitiveConstant::Int((lhs / rhs)?)),
            )),
            (lhs, rhs) => Err(RuntimeError::InvalidBinaryOperation(
                lhs,
                BinaryOperator::Times,
                rhs,
            )),
        }
    }
}

impl Rem for Constant {
    type Output = Result<Constant, RuntimeError>;

    fn rem(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (
                Constant::HashableConstant(HashableConstant::PrimitiveConstant(
                    PrimitiveConstant::Int(lhs),
                )),
                Constant::HashableConstant(HashableConstant::PrimitiveConstant(
                    PrimitiveConstant::Int(rhs),
                )),
            ) => Ok(Constant::HashableConstant(
                HashableConstant::PrimitiveConstant(PrimitiveConstant::Int((lhs % rhs)?)),
            )),
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
            (
                Constant::HashableConstant(HashableConstant::PrimitiveConstant(
                    PrimitiveConstant::Int(lhs),
                )),
                Constant::HashableConstant(HashableConstant::PrimitiveConstant(
                    PrimitiveConstant::Int(rhs),
                )),
            ) => Ok(Constant::HashableConstant(
                HashableConstant::PrimitiveConstant(PrimitiveConstant::Int((lhs - rhs)?)),
            )),
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
            (
                Constant::HashableConstant(HashableConstant::PrimitiveConstant(
                    PrimitiveConstant::Int(lhs),
                )),
                Constant::HashableConstant(HashableConstant::PrimitiveConstant(
                    PrimitiveConstant::Int(rhs),
                )),
            ) => Some(lhs.cmp(rhs)),

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
        BinaryOperator::Division => lhs / rhs,
        BinaryOperator::Times => lhs * rhs,
        BinaryOperator::Minus => lhs - rhs,
        BinaryOperator::Comparison(Comparison::Equal) => Ok((lhs == rhs).into()),
        BinaryOperator::Comparison(Comparison::Unequal) => Ok((lhs != rhs).into()),
        BinaryOperator::Comparison(cmp) => lhs
            .partial_cmp(&rhs)
            .ok_or(RuntimeError::InvalidBinaryOperation(lhs, op, rhs))
            .map(|ord| (cmp.evaluate(ord)).into()),
        BinaryOperator::Modulo => lhs % rhs,
        //TODO error on ty mismatch
        BinaryOperator::And => Ok((lhs.get_bool()? && rhs.get_bool()?).into()),
        BinaryOperator::Or => Ok((lhs.get_bool()? || rhs.get_bool()?).into()),
    }
}

impl From<bool> for Constant {
    fn from(value: bool) -> Self {
        Self::HashableConstant(HashableConstant::PrimitiveConstant(
            PrimitiveConstant::Bool(value),
        ))
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
    #[error("unexpected token '{0:?}' ({1})")]
    UnexpectedToken(Token, String),
    #[error("function '{0}' already defined")]
    AlreadyDefinedFunction(Word),
    #[error("function '{0}' not defined")]
    NotDefinedFunction(Word),
    #[error("module '{0}' not defined")]
    NotDefinedModule(Word),
    #[error("word '{0}' stands for multiple of variable, function, module")]
    NotUniquelyDefined(Word),
    //TODO more specific unified variants for these
    #[error("nothing with name '{0}' defined")]
    NotDefined(Word),
}

pub struct Tokens(Vec<Token>);

impl Tokens {
    fn get(&self) -> &Vec<Token> {
        &self.0
    }

    pub fn from_code(chars: &str, filename: PathBuf) -> Result<Self, SourceTokenizeError> {
        let mut vec = Vec::new();

        let mut p = Parsee::new(chars, filename);

        p.skip_whitespaces();

        while let Some(token) = p.parse_token()? {
            vec.push(token);
        }

        Ok(Self(vec))
    }

    pub fn get_top_level_declarations(
        &self,
    ) -> Result<(IndexSet<Word>, IndexSet<Word>), ParseError> {
        let mut stream = TokenStreamBasic::new(self);

        let mut fun_names = IndexSet::new();
        let mut module_names = IndexSet::new();

        loop {
            match stream.next() {
                // TODO: Error flow
                Err(_) => break,
                Ok(Token::Fun) => {
                    let name = stream.expect_word()?;
                    //TODO I don't like this clone.
                    let inserted = fun_names.insert(name.clone());
                    if !inserted {
                        return Err(ParseError::AlreadyDefinedFunction(name));
                    }
                }
                Ok(Token::Import) => {
                    let name = stream.expect_word()?;
                    //TODO I don't like this clone.
                    let inserted = module_names.insert(name.clone());
                    if !inserted {
                        return Err(ParseError::AlreadyDefinedFunction(name));
                    }
                }
                Ok(_) => {}
            }
        }

        Ok((fun_names, module_names))
    }
}

pub struct TokenStreamBasic<'a> {
    tokens: &'a Tokens,
    index: usize,
}

impl<'a> TokenStreamBasic<'a> {
    pub fn new(tokens: &'a Tokens) -> Self {
        Self { tokens, index: 0 }
    }

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

    fn expect_word(&mut self) -> Result<Word, ParseError> {
        match self.next()? {
            Token::Word(word) => Ok(word),
            token => Err(ParseError::UnexpectedToken(token, "expected a word".into())),
        }
    }
}

pub struct TokenStream<'a> {
    tokens: &'a Tokens,
    index: usize,
    #[expect(dead_code)]
    int_variables: IndexSet<Word>,
    stack: Stack,
    module_headers: &'a ModuleHeaders,
    module_id: ModuleId,
    expressions: &'a mut Expressions,
}

impl From<FunId> for Constant {
    fn from(value: FunId) -> Self {
        Constant::HashableConstant(HashableConstant::PrimitiveConstant(
            PrimitiveConstant::Callable(Callable::Fun(value)),
        ))
    }
}

//TODO pub
pub struct ModuleHeader(IndexMap<Word, ModuleOrFun>);

//TODO pub
pub struct ModuleHeaders(pub Vec<ModuleHeader>);

impl<'a> TokenStream<'a> {
    pub fn new(
        tokens: &'a Tokens,
        module_headers: &'a ModuleHeaders,
        module_id: ModuleId,
        expressions: &'a mut Expressions,
    ) -> Self {
        Self {
            tokens,
            index: 0,
            int_variables: IndexSet::new(),
            stack: Stack::new(),
            module_headers,
            module_id,
            expressions,
        }
    }

    fn parse_module_expr(&mut self, module_id: ModuleId) -> Result<Expr, ParseError> {
        self.expect(Token::DoubleColon)?;
        let entry = self.expect_word()?;

        match self.get_module_or_fun(module_id, &entry)? {
            ModuleOrFun::Module(module_id) => self.parse_module_expr(module_id),
            ModuleOrFun::Fun(fun_id) => Ok(Expr::Constant(fun_id.into())),
        }
    }

    fn get_module_or_fun(
        &self,
        module_id: ModuleId,
        entry: &Word,
    ) -> Result<ModuleOrFun, ParseError> {
        //TODO revisit expect
        self.module_headers
            .0
            .get(module_id.0)
            .expect("module_id came from us, I hope")
            .0
            .get(entry)
            .ok_or(ParseError::NotDefined(entry.clone()))
            .copied()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ModuleOrFun {
    Module(ModuleId),
    Fun(FunId),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Comparison {
    Smaller,
    Greater,
    Equal,
    Unequal,
    SmallerEqual,
    GreaterEqual,
}

impl Comparison {
    fn evaluate(&self, ord: Ordering) -> bool {
        match self {
            Comparison::Smaller => ord == Ordering::Less,
            Comparison::Greater => ord == Ordering::Greater,
            Comparison::Equal => ord == Ordering::Equal,
            Comparison::Unequal => ord != Ordering::Equal,
            Comparison::SmallerEqual => ord != Ordering::Greater,
            Comparison::GreaterEqual => ord != Ordering::Less,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOperator {
    Plus,
    Times,
    Division,
    Minus,
    Modulo,
    And,
    Or,
    Comparison(Comparison),
}

impl Display for Comparison {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Comparison::Smaller => write!(f, "<"),
            Comparison::Greater => write!(f, ">"),
            Comparison::Equal => write!(f, "=="),
            Comparison::Unequal => write!(f, "!="),
            Comparison::SmallerEqual => write!(f, "<="),
            Comparison::GreaterEqual => write!(f, ">="),
        }
    }
}

impl Display for BinaryOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryOperator::Plus => write!(f, "+"),
            BinaryOperator::Times => write!(f, "*"),
            BinaryOperator::Minus => write!(f, "-"),
            BinaryOperator::Modulo => write!(f, "%"),
            BinaryOperator::And => write!(f, "&&"),
            BinaryOperator::Or => write!(f, "||"),
            BinaryOperator::Division => write!(f, "/"),
            BinaryOperator::Comparison(cmp) => write!(f, "{cmp}"),
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
            BinaryOperator::Modulo => Stickyness::Multiplication,
            BinaryOperator::And => Stickyness::Conjunction,
            BinaryOperator::Or => Stickyness::Disjunction,
            BinaryOperator::Division => Stickyness::Multiplication,
            BinaryOperator::Comparison(_) => Stickyness::Comparison,
        }
    }
}

#[derive(Clone)]
enum FollowUp {
    BinaryOperator(BinaryOperator),
    End,
    Callable(Expr),
}
impl FollowUp {
    fn allow_assign(&self) -> bool {
        match self {
            FollowUp::BinaryOperator(_) | FollowUp::Callable(_) => true,
            FollowUp::End => false,
        }
    }
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

impl From<Callable> for Constant {
    fn from(value: Callable) -> Self {
        Constant::HashableConstant(HashableConstant::PrimitiveConstant(
            PrimitiveConstant::Callable(value),
        ))
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

    fn interpret_word(&mut self, word: &Word) -> Result<Expr, ParseError> {
        // //TODO Err flow
        let variable_id = self.stack.get(word).ok();
        let module_or_fun = self.get_module_or_fun(self.module_id, word).ok();
        let internal_fun = InternalFun::get(word);

        match (variable_id, module_or_fun, internal_fun) {
            (Some(variable_id), None, None) => {
                if self.entertain(Token::Assign) {
                    let expr = self.parse_expr()?;
                    return Ok(Expr::Assign(variable_id, self.expressions.add(expr)));
                }
                Ok(Expr::Variable(variable_id))
            }
            (None, Some(ModuleOrFun::Fun(fun_id)), None) => {
                Ok(Expr::Constant(Callable::Fun(fun_id).into()))
            }
            (None, Some(ModuleOrFun::Module(module_id)), None) => self.parse_module_expr(module_id),
            (None, None, Some(internal_fun)) => {
                Ok(Expr::Constant(Callable::InternalFun(internal_fun).into()))
            }
            (None, None, None) => Err(ParseError::NotDefined(word.clone())),
            (_, _, _) => Err(ParseError::NotUniquelyDefined(word.clone())),
        }
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
            Token::BinaryOperator(BinaryOperator::Minus) => {
                let inner_expr = self.parse_non_binary()?;
                Expr::Minus(self.expressions.add(inner_expr))
            }
            Token::GeneralTy(ty) => {
                let ty: Ty = ty.try_into()?;
                Expr::Constant(ty.into())
            }
            Token::Return => {
                let expr = if self.entertain(Token::Semicolon) {
                    Expr::Constant(Constant::HashableConstant(
                        HashableConstant::PrimitiveConstant(PrimitiveConstant::None),
                    ))
                } else {
                    self.parse_expr()?
                };
                Expr::Return(self.expressions.add(expr))
            }
            Token::Word(word) => {
                let expr = self.interpret_word(&word)?;
                if let Some(Token::ParanLeft) = self.peek() {
                    self.parse_fun_arguments(expr, None)?
                } else {
                    expr
                }
            }

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
            let callable_expr = self.interpret_word(&name)?;
            expr = self.parse_fun_arguments(callable_expr, Some(expr))?;
        }

        Ok(expr)
    }

    fn parse_follow_up(&mut self) -> Result<FollowUp, ParseError> {
        match self.peek() {
            Some(Token::BinaryOperator(op)) => {
                self.next().expect("just peeked");
                Ok(FollowUp::BinaryOperator(op))
            }
            Some(Token::Word(name)) => {
                self.next().expect("just peeked");
                //TODO where to convert to ExprId
                Ok(FollowUp::Callable(self.interpret_word(&name)?))
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

        if let Expr::Variable(variable_id) = ast {
            if follow_up.allow_assign() && self.entertain(Token::Assign) {
                let expr = self.parse_expr()?;
                let (range, lhs, rhs) = self.expressions.add_two(ast, expr);
                let expr = match follow_up {
                    FollowUp::BinaryOperator(binary_operator) => {
                        Expr::BinaryOp(lhs, binary_operator, rhs)
                    }
                    FollowUp::Callable(callable) => Expr::Call(Call {
                        callable: self.expressions.add(callable),
                        arguments: range,
                    }),
                    _ => unreachable!("allow_assign"),
                };

                let expr_id = self.expressions.add(expr);
                return Ok(Expr::Assign(variable_id, expr_id));
            }
        }

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

        let ty = if self.entertain(Token::Arrow) {
            self.expect_ty()?
        } else {
            GeneralTy::Ty(Ty::None)
        };

        for variable in variables.map.keys() {
            //TODO Test error
            self.stack.push(variable.clone())?;
        }

        let body = self.parse_block()?;

        self.stack.variables.clear();

        Ok(Fun::new(name, variables, ty, body))
    }

    fn expect_ty(&mut self) -> Result<GeneralTy, ParseError> {
        match self.next()? {
            Token::GeneralTy(word) => Ok(word),
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
            let expr = Expr::Constant(Constant::default());
            statements.push(expr);
        }

        let range: ExprRange = self.expressions.add_range(statements);

        Ok(Block::new(range))
    }

    fn parse_fun_arguments(
        &mut self,
        callable: Expr,
        leading: Option<Expr>,
    ) -> Result<Expr, ParseError> {
        let callable = self.expressions.add(callable);

        let (statements, _) = self.parse_expression_list_and_has_trailing_separator(
            Token::ParanLeft,
            Token::Comma,
            Token::ParanRight,
            false,
            leading,
        )?;

        let range = self.expressions.add_range(statements);

        Ok(Expr::Call(Call {
            callable,
            arguments: range,
        }))
    }
}

pub struct Module {
    ast: Ast,
}
impl Module {
    fn new(ast: Ast) -> Self {
        Self { ast }
    }
}

#[derive(Error, Debug)]
pub enum ModuleError {
    #[error("Tokenize Error: {0}")]
    TokenizeError(#[from] SourceTokenizeError),
    #[error("Parse Error: {0}")]
    ParseError(#[from] ParseError),
    #[error("Runtime Error: {0}")]
    RuntimeError(#[from] RuntimeError),
    #[error("Io Error for '{0}': {1}")]
    IoError(PathBuf, std::io::Error),
}

pub struct Modules {
    modules: Vec<Module>,
    expressions: Expressions,
    entry_point: Option<FunId>,
}

//TODO
fn _timed<F, Res>(atomic: &AtomicU64, fun: F) -> Res
where
    F: FnOnce() -> Res,
{
    let s = Instant::now();
    let res = fun();
    let e = Instant::now();

    let dur = (e - s).as_nanos() as u64;
    atomic.fetch_add(dur, std::sync::atomic::Ordering::Relaxed);

    res
}

impl Modules {
    fn new(modules: Vec<Module>, expressions: Expressions, entry_point: Option<FunId>) -> Self {
        Self {
            modules,
            expressions,
            entry_point,
        }
    }

    fn get_fun(&self, fun_id: FunId) -> &Fun {
        self.modules
            .get(fun_id.module_id.0)
            .expect("fun_id comes from us")
            .ast
            .funs
            .get(fun_id.fun_in_module)
            .expect("fun_id comes from us")
    }

    pub fn evaluate_main(&self) -> Result<Constant, RuntimeError> {
        let mut variable_values = VariableValues::new();

        self.evaluate_fun(
            self.entry_point.ok_or(RuntimeError::NoEntryPoint)?,
            Vec::new(),
            &mut variable_values,
        )
    }

    pub fn evaluate_fun(
        &self,
        fun_id: FunId,
        //TODO: This allocation slows us down.
        parameters: Vec<Constant>,
        variable_values: &mut VariableValues,
    ) -> Result<Constant, RuntimeError> {
        let fun = self.get_fun(fun_id);

        debug!("calling {}, parameters:", fun.name);

        if fun.variables.len() != parameters.len()
            || !fun
                .variables
                .map
                .iter()
                .zip(parameters.iter())
                .all(|((_, variable), constant)| constant.ty().satisfies(variable.general_ty))
        {
            return Err(RuntimeError::IncompatibleParameters(
                parameters.iter().map(|c| c.ty()).collect(),
                fun.variables.clone(),
                fun.name.clone(),
            ));
        }

        for (value, name) in parameters.iter().zip(fun.variables.map.keys()) {
            debug!("    {}: {}", name, value);
        }

        let old_reference = variable_values.reference;
        variable_values.reference = variable_values.values.len();

        for (id, parameter) in parameters.into_iter().enumerate() {
            variable_values.introduce(VariableId(id), parameter);
        }

        let result = self
            .evaluate_block(&fun.body, variable_values)?
            .unwrap_on_outer_layer();

        debug!("returning {} from {}", result, fun.name,);

        assert_eq!(
            variable_values.values.len() - variable_values.reference,
            fun.variables.len()
        );

        //TODO: Check when parsing fun
        if !result.ty().satisfies(fun.ty) {
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

    fn path_of(base_path: &Path, name: Word) -> PathBuf {
        base_path.with_file_name(format!("{}.ri", name.0))
    }

    fn evaluate_block_unscoped(
        &self,
        block: &Block,
        variable_values: &mut VariableValues,
    ) -> Result<Evaluation, RuntimeError> {
        if block.statements.start == block.statements.end {
            return Ok(Constant::default().into());
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

    fn evaluate_constant(&self, constant: Constant) -> Evaluation {
        constant.into()
    }

    fn evaluate_variable(
        &self,
        variable_id: &VariableId,
        variable_values: &mut VariableValues,
    ) -> Result<Evaluation, RuntimeError> {
        Ok(variable_values.get(variable_id)?.into())
    }

    fn evaluate_binary_op(
        &self,
        lhs_expr: &ExprId,
        op: &BinaryOperator,
        rhs_expr: &ExprId,
        variable_values: &mut VariableValues,
    ) -> Result<Evaluation, RuntimeError> {
        let lhs = self.evaluate_expr(*lhs_expr, variable_values)?;
        let Some(lhs) = lhs.some_or_please_return() else {
            return Ok(lhs);
        };

        // Short circuiting
        if op == &BinaryOperator::And && !lhs.get_bool()? {
            let res: Constant = false.into();
            return Ok(res.into());
        }

        if op == &BinaryOperator::Or && lhs.get_bool()? {
            let res: Constant = true.into();
            return Ok(res.into());
        }

        let rhs = self.evaluate_expr(*rhs_expr, variable_values)?;
        let Some(rhs) = rhs.some_or_please_return() else {
            return Ok(rhs);
        };

        let combined = combine_with_operator(lhs.clone(), *op, rhs.clone());

        Ok(combined?.into())
    }

    fn evaluate_return(
        &self,
        expr: &ExprId,
        variable_values: &mut VariableValues,
    ) -> Result<Evaluation, RuntimeError> {
        self.evaluate_expr(*expr, variable_values)
            .map(|evaluation| evaluation.returning())
    }

    fn evaluate_if(
        &self,
        expr: &ExprId,
        if_block: &Block,
        else_expr: &ExprId,
        variable_values: &mut VariableValues,
    ) -> Result<Evaluation, RuntimeError> {
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

    fn evaluate_call(
        &self,
        call: &Call,
        variable_values: &mut VariableValues,
    ) -> Result<Evaluation, RuntimeError> {
        let callable = self.evaluate_expr(call.callable, variable_values)?;
        let Some(callable) = callable.some_or_please_return() else {
            return Ok(callable);
        };

        let &Constant::HashableConstant(HashableConstant::PrimitiveConstant(
            PrimitiveConstant::Callable(callable),
        )) = callable
        else {
            return Err(RuntimeError::NotCallable(callable.clone()));
        };

        let mut parameters = Vec::new();
        for id in call.arguments.start.0..call.arguments.end.0 {
            let eval = self.evaluate_expr(ExprId(id), variable_values)?;
            let Some(parameter) = eval.some_or_please_return() else {
                return Ok(eval);
            };

            parameters.push(parameter.clone());
        }

        self.call(callable, parameters, variable_values)
            .map(|c| c.into())
    }

    fn evaluate_assign(
        &self,
        variable_id: &VariableId,
        expr_id: &ExprId,
        variable_values: &mut VariableValues,
    ) -> Result<Evaluation, RuntimeError> {
        let eval = self.evaluate_expr(*expr_id, variable_values)?;
        let Some(eval) = eval.some_or_please_return() else {
            return Ok(eval);
        };
        variable_values.set(*variable_id, eval.clone())?;
        Ok(Constant::default().into())
    }

    //TODO: Can we combine with assign?
    fn evaluate_introduce(
        &self,
        variable_id: &VariableId,
        expr_id: &ExprId,
        variable_values: &mut VariableValues,
    ) -> Result<Evaluation, RuntimeError> {
        //SYNC(INTRO_AFTER_EVAL) Variable gets introduced after evaluating its initial value.
        let eval = self.evaluate_expr(*expr_id, variable_values)?;
        let Some(eval) = eval.some_or_please_return() else {
            return Ok(eval);
        };
        variable_values.introduce(*variable_id, eval.clone());
        Ok(Constant::default().into())
    }

    fn evaluate_while(
        &self,
        expr_id: &ExprId,
        block: &Block,
        variable_values: &mut VariableValues,
    ) -> Result<Evaluation, RuntimeError> {
        loop {
            let eval = self.evaluate_expr(*expr_id, variable_values)?;
            let Some(constant) = eval.some_or_please_return() else {
                return Ok(eval);
            };

            let b = constant.get_bool()?;

            if !b {
                return Ok(Constant::default().into());
            }

            let block_result = self.evaluate_block(block, variable_values)?;
            let Some(_) = block_result.some_or_please_return() else {
                return Ok(block_result);
            };
        }
    }

    fn evaluate_for(
        &self,
        var0: &VariableId,
        var1: &Option<VariableId>,
        iterable: &ExprId,
        block: &Block,
        variable_values: &mut VariableValues,
    ) -> Result<Evaluation, RuntimeError> {
        let iterable = self.evaluate_expr(*iterable, variable_values)?;
        let Some(iterable) = iterable.some_or_please_return() else {
            return Ok(iterable);
        };

        let iter = match iterable {
            Constant::HashableConstant(HashableConstant::List(rc)) => rc.borrow().clone(),
            _ => return Err(RuntimeError::NotIterable(iterable.clone())),
        };

        match var1 {
            Some(var1) => {
                for (i, c) in iter.into_iter().enumerate() {
                    variable_values.introduce(*var0, (i as i128).into());
                    variable_values.introduce(*var1, c.into());

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
                    variable_values.introduce(*var0, c.into());

                    let eval = self.evaluate_block(block, variable_values)?;

                    variable_values.values.pop();

                    let Some(_) = eval.some_or_please_return() else {
                        return Ok(eval);
                    };
                }
            }
        }

        Ok(Constant::default().into())
    }

    fn evaluate_negate(
        &self,
        expr_id: &ExprId,
        variable_values: &mut VariableValues,
    ) -> Result<Evaluation, RuntimeError> {
        let inner_value = self.evaluate_expr(*expr_id, variable_values)?;
        let Some(inner_value) = inner_value.some_or_please_return() else {
            return Ok(inner_value);
        };

        let value: Constant = (!inner_value.get_bool()?).into();
        Ok(value.into())
    }

    fn evaluate_minus(
        &self,
        expr_id: &ExprId,
        variable_values: &mut VariableValues,
    ) -> Result<Evaluation, RuntimeError> {
        let inner_value = self.evaluate_expr(*expr_id, variable_values)?;
        let Some(inner_value) = inner_value.some_or_please_return() else {
            return Ok(inner_value);
        };

        let value: Constant = (-inner_value.get_int()?).into();
        Ok(value.into())
    }

    fn evaluate_expr(
        &self,
        expr_id: ExprId,
        variable_values: &mut VariableValues,
    ) -> Result<Evaluation, RuntimeError> {
        match self.expressions.get(expr_id) {
            Expr::Constant(constant) => Ok(self.evaluate_constant(constant.clone())),
            Expr::Variable(variable_id) => self.evaluate_variable(variable_id, variable_values),
            Expr::BinaryOp(lhs_expr, op, rhs_expr) => {
                self.evaluate_binary_op(lhs_expr, op, rhs_expr, variable_values)
            }
            Expr::Block(block) => self.evaluate_block(block, variable_values),
            Expr::Return(expr) => self.evaluate_return(expr, variable_values),
            Expr::If(expr, if_block, else_expr) => {
                self.evaluate_if(expr, if_block, else_expr, variable_values)
            }
            Expr::Call(call) => self.evaluate_call(call, variable_values),
            Expr::Assign(variable_id, expr_id) => {
                self.evaluate_assign(variable_id, expr_id, variable_values)
            }
            Expr::Introduce(variable_id, expr_id) => {
                self.evaluate_introduce(variable_id, expr_id, variable_values)
            }
            Expr::While(expr_id, block) => self.evaluate_while(expr_id, block, variable_values),
            Expr::For(var0, var1, iterable, block) => {
                self.evaluate_for(var0, var1, iterable, block, variable_values)
            }
            Expr::Negate(expr_id) => self.evaluate_negate(expr_id, variable_values),
            Expr::Minus(expr_id) => self.evaluate_minus(expr_id, variable_values),
        }
    }

    fn call(
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

    pub fn evaluate_internal_fun(
        &self,
        internal_fun: InternalFun,
        parameters: Vec<Constant>,
        variable_values: &mut VariableValues,
    ) -> Result<Constant, RuntimeError> {
        match internal_fun {
            InternalFun::New => {
                if let Some(&Constant::HashableConstant(HashableConstant::PrimitiveConstant(
                    PrimitiveConstant::Ty(ty),
                ))) = parameters.iter().collect_single()
                {
                    match ty {
                        Ty::List => {
                            return Ok(Constant::HashableConstant(HashableConstant::List(
                                Default::default(),
                            )))
                        }
                        Ty::Dict => return Ok(Constant::Dict(Default::default())),
                        _ => return Err(RuntimeError::StaticFunctionOnWrongType(ty, internal_fun)),
                    }
                }
            }
            InternalFun::Push => {
                if let Some((
                    Constant::HashableConstant(HashableConstant::List(list)),
                    Constant::HashableConstant(c),
                )) = parameters.iter().collect_tuple()
                {
                    (*list).borrow_mut().push(c.clone());
                    return Ok(Constant::default());
                }
            }
            InternalFun::Get => {
                if let Some((
                    Constant::HashableConstant(HashableConstant::List(list)),
                    &Constant::HashableConstant(HashableConstant::PrimitiveConstant(
                        PrimitiveConstant::Int(i),
                    )),
                )) = parameters.iter().collect_tuple()
                {
                    let id: usize = i.try_into().map_err(|_| {
                        RuntimeError::IndexOutOfBounds(list.borrow().clone().into(), i)
                    })?;

                    let Some(result) = list.borrow().get(id).cloned() else {
                        return Err(RuntimeError::IndexOutOfBounds(
                            list.borrow().clone().into(),
                            i,
                        ));
                    };
                    return Ok(result.into());
                }

                if let Some((Constant::Dict(dict), Constant::HashableConstant(key))) =
                    parameters.iter().collect_tuple()
                {
                    let Some(result) = dict.borrow().get(key).cloned() else {
                        return Err(RuntimeError::KeyDoesNotExist(key.clone()));
                    };
                    return Ok(result);
                }
            }

            InternalFun::Pop => {
                if let Some(Constant::HashableConstant(HashableConstant::List(list))) =
                    parameters.iter().collect_single()
                {
                    let Some(res) = list.borrow_mut().pop() else {
                        return Err(RuntimeError::InternalOperationInvalid(
                            internal_fun,
                            parameters.clone(),
                        ));
                    };

                    return Ok(res.into());
                }
            }
            InternalFun::Len => {
                if let Some(Constant::HashableConstant(HashableConstant::List(list))) =
                    parameters.iter().collect_single()
                {
                    //TODO Cast so okay?
                    return Ok((list.borrow().len() as i128).into());
                }

                if let Some(Constant::Dict(dict)) = parameters.iter().collect_single() {
                    //TODO Cast so okay?
                    return Ok((dict.borrow().len() as i128).into());
                }
            }
            InternalFun::FromFile => {
                if let Some((
                    &Constant::HashableConstant(HashableConstant::PrimitiveConstant(
                        PrimitiveConstant::Ty(ty),
                    )),
                    Constant::HashableConstant(HashableConstant::List(list)),
                )) = parameters.iter().collect_tuple()
                {
                    match ty {
                        Ty::List => {
                            let path = try_list_to_string(&list.borrow())?;

                            let file_content =
                                read_to_string(&path).map_err(|err| RuntimeError::Io(err, path))?;
                            let list: Vec<HashableConstant> =
                                file_content.chars().map(|c| c.into()).collect();
                            return Ok(list.into());
                        }
                        _ => return Err(RuntimeError::StaticFunctionOnWrongType(ty, internal_fun)),
                    }
                }
            }
            InternalFun::SplitWhitespace => {
                if let Some(Constant::HashableConstant(HashableConstant::List(list))) =
                    parameters.iter().collect_single()
                {
                    let string = try_list_to_string(&list.borrow())?;
                    let result: Vec<HashableConstant> = string
                        .split_whitespace()
                        .map(|sub| str_to_list(sub).into())
                        .collect_vec();

                    return Ok(result.into());
                }
            }
            InternalFun::Parse => {
                if let Some((
                    Constant::HashableConstant(HashableConstant::List(list)),
                    Constant::HashableConstant(HashableConstant::PrimitiveConstant(
                        PrimitiveConstant::Ty(ty),
                    )),
                )) = parameters.iter().collect_tuple()
                {
                    let string = try_list_to_string(&list.borrow())?;

                    let res = match ty {
                        Ty::Int => {
                            let i: i128 = string
                                .parse()
                                .map_err(|err| RuntimeError::Parse(string, Ty::Int, err))?;
                            i.into()
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
                if let Some(Constant::HashableConstant(HashableConstant::List(list))) =
                    parameters.iter().collect_single()
                {
                    let ty = list.borrow().first().map(|elem| elem.ty());
                    match ty {
                        Some(Ty::Int) => {
                            let mut new_list = Vec::new();
                            for val in list.borrow().iter() {
                                let &HashableConstant::PrimitiveConstant(PrimitiveConstant::Int(i)) =
                                    val
                                else {
                                    return Err(RuntimeError::TypeError(Ty::Int, val.ty()));
                                };
                                new_list.push(i);
                            }
                            new_list.sort();
                            let new_list = new_list
                                .into_iter()
                                .map(|i| {
                                    HashableConstant::PrimitiveConstant(PrimitiveConstant::Int(i))
                                })
                                .collect();
                            *list.borrow_mut() = new_list;
                        }
                        Some(Ty::List) => {
                            let mut new_list = Vec::new();

                            for val in list.borrow().iter() {
                                let HashableConstant::List(elem) = val else {
                                    return Err(RuntimeError::TypeError(Ty::List, val.ty()));
                                };

                                let mut s = String::new();
                                for c in elem.borrow().iter() {
                                    let &HashableConstant::PrimitiveConstant(
                                        PrimitiveConstant::Char(ch),
                                    ) = c
                                    else {
                                        return Err(RuntimeError::TypeError(Ty::Char, c.ty()));
                                    };

                                    s.push(ch);
                                }
                                new_list.push(s);
                            }
                            new_list.sort();
                            let new_list = new_list
                                .into_iter()
                                .map(|i| {
                                    let i = i
                                        .chars()
                                        .map(|c| {
                                            HashableConstant::PrimitiveConstant(
                                                PrimitiveConstant::Char(c),
                                            )
                                        })
                                        .collect_vec();
                                    HashableConstant::List(Rc::new(RefCell::new(i)))
                                })
                                .collect();
                            *list.borrow_mut() = new_list;
                        }
                        Some(_) => todo!(),
                        None => {}
                    }

                    return Ok(Constant::default());
                }
            }
            InternalFun::Abs => {
                if let Some(Constant::HashableConstant(HashableConstant::PrimitiveConstant(
                    PrimitiveConstant::Int(i),
                ))) = parameters.iter().collect_single()
                {
                    let IntConstant::Small(i) = i;
                    return Ok(i.abs().into());
                }
            }
            InternalFun::Lines => {
                if let Some(Constant::HashableConstant(HashableConstant::List(list))) =
                    parameters.iter().collect_single()
                {
                    let string = try_list_to_string(&list.borrow())?;
                    let result: Vec<HashableConstant> = string
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
                if let Some(&Constant::HashableConstant(HashableConstant::PrimitiveConstant(
                    PrimitiveConstant::Callable(callable),
                ))) = params_iter.next()
                {
                    //TODO Collect meh
                    return self.call(callable, params_iter.cloned().collect(), variable_values);
                }
            }
            InternalFun::SetNew => {
                if let Some((Constant::Dict(dict), Constant::HashableConstant(key), value)) =
                    parameters.iter().collect_tuple()
                {
                    //TODO What if key is mutated? Does this break indexmap invariants?
                    match dict.borrow_mut().entry(key.clone()) {
                        indexmap::map::Entry::Occupied(_) => {
                            return Err(RuntimeError::KeyAlreadyExists(key.clone()))
                        }
                        indexmap::map::Entry::Vacant(vacant_entry) => {
                            vacant_entry.insert(value.clone())
                        }
                    };

                    return Ok(Constant::default());
                }
            }
            InternalFun::Set => {
                if let Some((
                    Constant::HashableConstant(HashableConstant::List(list)),
                    &Constant::HashableConstant(HashableConstant::PrimitiveConstant(
                        PrimitiveConstant::Int(i),
                    )),
                    Constant::HashableConstant(value),
                )) = parameters.iter().collect_tuple()
                {
                    let id: usize = i.try_into().map_err(|_| {
                        RuntimeError::IndexOutOfBounds(list.borrow().clone().into(), i)
                    })?;

                    let mut list_borrow = list.borrow_mut();

                    list_borrow
                        .get_mut(id)
                        .map(|entry| {
                            *entry = value.clone();
                        })
                        .ok_or(RuntimeError::IndexOutOfBounds(
                            list_borrow.clone().into(),
                            i,
                        ))?;

                    return Ok(Constant::default());
                }
            }
            InternalFun::Has => {
                if let Some((Constant::Dict(dict), Constant::HashableConstant(key))) =
                    parameters.iter().collect_tuple()
                {
                    return Ok(dict.borrow().contains_key(key).into());
                }
            }
            InternalFun::Keys => {
                if let Some(Constant::Dict(dict)) = parameters.iter().collect_single() {
                    return Ok(dict.borrow().keys().cloned().collect_vec().into());
                }
            }
            InternalFun::DeepClone => {
                if let Some(c) = parameters.iter().collect_single() {
                    return Ok(c.deep_clone());
                }
            }
            InternalFun::Remove => {
                if let Some((Constant::Dict(dict), Constant::HashableConstant(key))) =
                    parameters.iter().collect_tuple()
                {
                    let Some((_, result)) = dict.borrow_mut().swap_remove_entry(key) else {
                        return Err(RuntimeError::KeyDoesNotExist(key.clone()));
                    };

                    return Ok(result);
                }
            }
            InternalFun::UpdateExisting => {
                if let Some((Constant::Dict(dict), Constant::HashableConstant(key), value)) =
                    parameters.iter().collect_tuple()
                {
                    let mut dict_borrow = dict.borrow_mut();

                    match dict_borrow.entry(key.clone()) {
                        indexmap::map::Entry::Occupied(mut occupied_entry) => {
                            *occupied_entry.get_mut() = value.clone()
                        }
                        indexmap::map::Entry::Vacant(_) => {
                            return Err(RuntimeError::KeyDoesNotExist(key.clone()))
                        }
                    }

                    return Ok(Constant::default());
                }
            }
            InternalFun::Xor => {
                if let Some((
                    Constant::HashableConstant(HashableConstant::PrimitiveConstant(
                        PrimitiveConstant::Int(i),
                    )),
                    Constant::HashableConstant(HashableConstant::PrimitiveConstant(
                        PrimitiveConstant::Int(j),
                    )),
                )) = parameters.iter().collect_tuple()
                {
                    let IntConstant::Small(i) = i;
                    let IntConstant::Small(j) = j;
                    return Ok((i ^ j).into());
                }
            }
            InternalFun::Intersection => {
                if let Some((Constant::Dict(set1), Constant::Dict(set2))) =
                    parameters.iter().collect_tuple()
                {
                    let set1 = set1.borrow();
                    let set2 = set2.borrow();

                    //TODO: Proper error.
                    assert!(set1.values().all(|v| *v == Default::default()));
                    assert!(set2.values().all(|v| *v == Default::default()));

                    let keys1: IndexSet<_> = set1.keys().collect();
                    let keys2: IndexSet<_> = set2.keys().collect();

                    let mut result = IndexMap::new();
                    for key in keys1.intersection(&keys2) {
                        result.insert((*key).clone(), Constant::default());
                    }
                    return Ok(Constant::Dict(Rc::new(RefCell::new(result))));
                }
            }
        }

        Err(RuntimeError::IncompatibleParametersForInternal(
            parameters.iter().map(|c| c.ty()).collect(),
            internal_fun,
        ))
    }

    pub fn from_entry_file(path_of_entry_module: &Path) -> Result<Self, ModuleError> {
        let mut path_to_module_id: IndexSet<PathBuf> = indexset! { path_of_entry_module.into() };
        let mut module_headers = Vec::new();
        let mut tokens_vec = Vec::new();

        while let Some(path_of_current_module) =
            path_to_module_id.get_index(module_headers.len()).cloned()
        {
            let code = read_to_string(path_of_current_module.clone())
                .map_err(|err| ModuleError::IoError(path_of_current_module.clone(), err))?;

            let tokens = Tokens::from_code(&code, path_of_current_module.clone())?;

            let (fun_names, module_names) = tokens.get_top_level_declarations()?;

            let mut map = IndexMap::new();
            for (i, fun_name) in fun_names.into_iter().enumerate() {
                map.insert(
                    fun_name,
                    ModuleOrFun::Fun(FunId {
                        module_id: ModuleId(module_headers.len()),
                        fun_in_module: i,
                    }),
                );
            }
            for module_name in module_names {
                let path_of_referenced_module =
                    Self::path_of(&path_of_current_module, module_name.clone());
                let (i, _) = path_to_module_id.insert_full(path_of_referenced_module);
                map.insert(module_name, ModuleOrFun::Module(ModuleId(i)));
            }

            module_headers.push(ModuleHeader(map));
            tokens_vec.push(tokens);
        }

        let module_headers = ModuleHeaders(module_headers);

        let mut expressions = Expressions::new();
        let mut modules = Vec::new();

        for (i, tokens) in tokens_vec.into_iter().enumerate() {
            let module_id = ModuleId(i);
            modules.push(tokens.parse(&module_headers, module_id, &mut expressions)?);
        }

        let entry_point = module_headers
            .0
            .first()
            .expect("entry module")
            .0
            .get(&*MAIN)
            .and_then(|module_or_fun| match module_or_fun {
                ModuleOrFun::Module(_) => None,
                ModuleOrFun::Fun(fun_id) => Some(fun_id),
            })
            .copied();

        Ok(Modules::new(modules, expressions, entry_point))
    }
}

impl Tokens {
    pub fn parse(
        &self,
        module_headers: &ModuleHeaders,
        module_id: ModuleId,
        expressions: &mut Expressions,
    ) -> Result<Module, ParseError> {
        let mut funs = Vec::new();

        let mut ts = TokenStream::new(self, module_headers, module_id, expressions);

        loop {
            match ts.peek() {
                None => break,
                Some(Token::Fun) => {
                    let fun: Fun = ts.parse_fun()?;
                    assert_eq!(
                        module_headers
                            .0
                            .get(module_id.0)
                            .expect("own id should be there")
                            .0
                            .get(&fun.name),
                        Some(&ModuleOrFun::Fun(FunId::new(module_id, funs.len())))
                    );
                    funs.push(fun);
                }
                Some(Token::Import) => {
                    ts.next().expect("just peeked");
                    ts.expect_word()?;
                    ts.expect(Token::Semicolon)?;
                }
                Some(unexpected) => {
                    return Err(ParseError::UnexpectedToken(
                        unexpected,
                        format!(
                            "expected a top level declaration (currently {} or {}))",
                            Token::Fun,
                            Token::Import
                        ),
                    ))
                }
            }
        }

        // This should be implicit by the above code.
        assert!(ts.is_fully_parsed());

        //TODO: Verify that there are no arguments for main?

        Ok(Module::new(Ast::new(funs)))
    }
}

pub struct Expressions {
    vec: Vec<Expr>,
}
impl Expressions {
    pub fn new() -> Self {
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

    fn add_two(&mut self, lhs: Expr, rhs: Expr) -> (ExprRange, ExprId, ExprId) {
        let first = self.add(lhs);
        let second = self.add(rhs);
        (
            ExprRange {
                start: first,
                end: self.next_expr_id(),
            },
            first,
            second,
        )
    }
}

impl Default for Expressions {
    fn default() -> Self {
        Self::new()
    }
}

impl From<Ty> for Constant {
    fn from(value: Ty) -> Self {
        Constant::HashableConstant(HashableConstant::PrimitiveConstant(PrimitiveConstant::Ty(
            value,
        )))
    }
}

impl From<i128> for Constant {
    fn from(value: i128) -> Self {
        Constant::HashableConstant(HashableConstant::PrimitiveConstant(PrimitiveConstant::Int(
            IntConstant::Small(value),
        )))
    }
}

impl From<char> for HashableConstant {
    fn from(value: char) -> Self {
        HashableConstant::PrimitiveConstant(PrimitiveConstant::Char(value))
    }
}

impl From<HashableConstant> for Constant {
    fn from(value: HashableConstant) -> Self {
        Constant::HashableConstant(value)
    }
}
impl From<Vec<HashableConstant>> for HashableConstant {
    fn from(value: Vec<HashableConstant>) -> Self {
        HashableConstant::List(Rc::new(RefCell::new(value)))
    }
}

pub struct Ast {
    funs: Vec<Fun>,
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
    fn new(funs: Vec<Fun>) -> Self {
        Self { funs }
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

    let main = Fun::new(name.clone(), Variables::new(), GeneralTy::Ty(ty), block);

    let main_id = FunId::new(ModuleId(0), 0);

    let ast = Ast::new(vec![main]);

    let module = Module::new(ast);

    let modules = Modules {
        modules: vec![module],
        expressions,
        entry_point: Some(main_id),
    };

    modules.evaluate_main()
}

static MAIN: LazyLock<Word> = LazyLock::new(|| "main".try_into().expect("valid word"));

fn try_list_to_string(list: &Vec<HashableConstant>) -> Result<String, RuntimeError> {
    let mut s = String::new();

    for c in list {
        let HashableConstant::PrimitiveConstant(PrimitiveConstant::Char(c)) = c else {
            //TODO test case for that; maybe better error message due to context
            return Err(RuntimeError::TypeError(Ty::Char, c.ty()));
        };

        s.push(*c);
    }

    Ok(s)
}

fn str_to_list(s: &str) -> Vec<HashableConstant> {
    s.chars()
        .map(|c| HashableConstant::PrimitiveConstant(PrimitiveConstant::Char(c)))
        .collect()
}

#[derive(Debug)]
pub struct Fun {
    name: Word,
    variables: Variables,
    ty: GeneralTy,
    body: Block,
}
impl Fun {
    fn new(name: Word, variables: Variables, ty: GeneralTy, body: Block) -> Self {
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
    general_ty: GeneralTy,
}
impl Variable {
    fn new(name: Word, general_ty: GeneralTy) -> Self {
        Self { name, general_ty }
    }
}

impl From<char> for Constant {
    fn from(value: char) -> Self {
        Constant::HashableConstant(HashableConstant::PrimitiveConstant(
            PrimitiveConstant::Char(value),
        ))
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
    filename: PathBuf,
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

    fn new(chars: &str, filename: PathBuf) -> Self {
        let lines = chars
            .lines()
            .map(|line| line.chars().chain(once('\n')).collect())
            .collect();
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
            //TODO Fails with "Done"
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
                return Ok(Some(Token::BinaryOperator(BinaryOperator::Comparison(
                    Comparison::Equal,
                ))));
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
                return Ok(Some(Token::BinaryOperator(BinaryOperator::Comparison(
                    Comparison::Unequal,
                ))));
            }

            if matches!(
                token,
                Token::BinaryOperator(BinaryOperator::Comparison(Comparison::Greater))
            ) && Some('=') == self.peek()
            {
                self.next();
                return Ok(Some(Token::BinaryOperator(BinaryOperator::Comparison(
                    Comparison::GreaterEqual,
                ))));
            }

            if matches!(
                token,
                Token::BinaryOperator(BinaryOperator::Comparison(Comparison::Smaller))
            ) && Some('=') == self.peek()
            {
                self.next();
                return Ok(Some(Token::BinaryOperator(BinaryOperator::Comparison(
                    Comparison::SmallerEqual,
                ))));
            }

            if matches!(token, Token::Colon) && Some(':') == self.peek() {
                self.next();
                return Ok(Some(Token::DoubleColon));
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

        let i: i128 = word.parse().unwrap();

        Ok(Token::Literal(i.into()))
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

        Ok(Token::Literal(c.into()))
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

            list.push(c.into());
        }
    }
}

#[derive(Debug, Error)]
pub struct SourceError<E: Error> {
    line_id: usize,
    character_id: usize,
    filename: PathBuf,
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
            "{:?}:{}:{}\n{}\n{}\n{}",
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
    fn new(filename: PathBuf, line_id: usize, character_id: usize, line: String, error: E) -> Self {
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
        '/' => Some(Token::BinaryOperator(BinaryOperator::Division)),
        '-' => Some(Token::BinaryOperator(BinaryOperator::Minus)),
        '*' => Some(Token::BinaryOperator(BinaryOperator::Times)),
        '<' => Some(Token::BinaryOperator(BinaryOperator::Comparison(
            Comparison::Smaller,
        ))),
        '>' => Some(Token::BinaryOperator(BinaryOperator::Comparison(
            Comparison::Greater,
        ))),
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
        Some(Token::GeneralTy(GeneralTy::Ty(Ty::List)))
    } else if keyword == "Dict" {
        Some(Token::GeneralTy(GeneralTy::Ty(Ty::Dict)))
    } else if keyword == "Range" {
        Some(Token::GeneralTy(GeneralTy::Ty(Ty::Range)))
    } else if keyword == "int" {
        Some(Token::GeneralTy(GeneralTy::Ty(Ty::Int)))
    } else if keyword == "TYPE" {
        Some(Token::GeneralTy(GeneralTy::Ty(Ty::Ty)))
    } else if keyword == "None" {
        Some(Token::Literal(Default::default()))
    } else if keyword == "import" {
        Some(Token::Import)
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
        Some(Token::GeneralTy(GeneralTy::Ty(Ty::Bool)))
    } else if keyword == "any" {
        Some(Token::GeneralTy(GeneralTy::Any))
    } else if keyword == "char" {
        Some(Token::GeneralTy(GeneralTy::Ty(Ty::Char)))
    } else if keyword == "Callable" {
        Some(Token::GeneralTy(GeneralTy::Ty(Ty::Callable)))
    } else if keyword == "true" {
        Some(Token::Literal(true.into()))
    } else if keyword == "false" {
        Some(Token::Literal(false.into()))
    } else {
        None
    }
}

impl From<Vec<HashableConstant>> for Constant {
    fn from(value: Vec<HashableConstant>) -> Self {
        Self::HashableConstant(HashableConstant::List(Rc::new(RefCell::new(value))))
    }
}

// TODO pub
pub mod word {
    use std::fmt::Display;

    use super::TokenizeError;

    #[derive(Hash, PartialEq, Eq, Debug, Clone)]
    // TODO private
    pub struct Word(pub String);

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
    Dict,
    List,
    Range,
    Ty,
    Callable,
    Char,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum GeneralTy {
    Ty(Ty),
    Any,
}

impl Display for GeneralTy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeneralTy::Ty(ty) => write!(f, "{ty}"),
            GeneralTy::Any => write!(f, "any"),
        }
    }
}

impl Ty {
    fn satisfies(self, other: GeneralTy) -> bool {
        match (self, other) {
            (_, GeneralTy::Any) => true,
            (lhs, GeneralTy::Ty(rhs)) => lhs == rhs,
        }
    }
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
            Ty::Dict => write!(f, "Dict"),
        }
    }
}

impl TryFrom<GeneralTy> for Ty {
    type Error = ParseError;

    fn try_from(value: GeneralTy) -> Result<Self, Self::Error> {
        match value {
            GeneralTy::Ty(ty) => Ok(ty),
            GeneralTy::Any => todo!(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Token {
    Return,
    Else,
    Import,
    DoubleColon,
    GeneralTy(GeneralTy),
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
            Token::GeneralTy(ty) => write!(f, "{ty}"),
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
            Token::Import => write!(f, "import"),
            Token::DoubleColon => write!(f, "::"),
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
