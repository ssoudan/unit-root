//! DF benchmark
#![allow(missing_docs)]
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::SeedableRng;
use unit_root::prelude::tools::dickeyfuller;
use unit_root::utils::gen_ar_1;

fn df_benchmark(c: &mut Criterion) {
    for size in [100, 200, 500, 1000, 5000].iter() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        let mu = 0.;
        let delta = 0.5;
        let sigma = 1.0;
        let y = gen_ar_1(&mut rng, *size, mu, delta, sigma);

        c.bench_with_input(BenchmarkId::new("df", size), &y, |b, y| {
            b.iter(|| dickeyfuller::constant_no_trend_test(y))
        });
    }
}

criterion_group!(benches, df_benchmark);
criterion_main!(benches);
