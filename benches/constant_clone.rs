use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

fn criterion_benchmark(c: &mut Criterion) {
    let inner = Inner::default();
    let inner2 = Inner2::default();

    let mut group = c.benchmark_group("copy slowdown");

    group.bench_function("with copy", |b| b.iter(|| *black_box(&inner)));
    group.bench_function("without copy", |b| b.iter(|| black_box(&inner2).clone()));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

#[derive(Clone, Copy)]
pub enum Inner {
    Int(i64),
    Bool(bool),
}

impl Default for Inner {
    fn default() -> Self {
        Self::Bool(false)
    }
}

impl Default for Inner2 {
    fn default() -> Self {
        Self::Bool(false)
    }
}

#[derive(Clone)]
pub enum Inner2 {
    Int(i64),
    Bool(bool),
}
