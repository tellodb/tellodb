use criterion::{criterion_group, criterion_main, Criterion};

fn empty_benchmark(c: &mut Criterion) {
    c.bench_function("placeholder", |b| b.iter(|| std::hint::black_box(2 + 2)));
}

criterion_group!(benches, empty_benchmark);
criterion_main!(benches);
