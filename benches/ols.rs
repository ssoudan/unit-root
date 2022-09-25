//! OLS benchmark
#![allow(missing_docs)]
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use unit_root::regression::ols;
use unit_root::utils::gen_affine_data;

fn ols_benchmark(c: &mut Criterion) {
    for size in [100, 200, 500, 1000, 5000].iter() {
        let mu = 0.;
        let beta = 0.5;
        let (x, y) = gen_affine_data(*size, mu, beta);

        c.bench_with_input(BenchmarkId::new("ols", size), &(y, x), |b, yx| {
            let (y, x) = yx;

            b.iter(|| ols(y, x))
        });
    }
}

criterion_group!(benches, ols_benchmark);
criterion_main!(benches);
