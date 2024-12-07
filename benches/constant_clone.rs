use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

fn criterion_benchmark(c: &mut Criterion) {
    let copyable = Copyable::Bool(false);
    let not_copyable = NotCopyable::Bool(false);

    let mut group = c.benchmark_group("copy slowdown");

    group.bench_function("copy", |b| {
        b.iter(|| black_box(black_box(&copyable).clone()))
    });
    group.bench_function("no copy", |b| {
        b.iter(|| black_box(black_box(&not_copyable).clone()))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

#[derive(Clone, Copy)]
pub enum Copyable {
    Int(i64),
    Bool(bool),
}

#[derive(Clone)]
pub enum NotCopyable {
    Int(i64),
    Bool(bool),
}
