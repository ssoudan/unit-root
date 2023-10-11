// Copyright (c) 2022. Sebastien Soudan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http:www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! ADF benchmark
#![allow(missing_docs)]
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::SeedableRng;
use unit_root::prelude::distrib::Regression;
use unit_root::prelude::tools::adf::adf_test;
use unit_root::utils::gen_ar_1;

fn adf_benchmark_f32_2(c: &mut Criterion) {
    for size in [100, 200, 500, 1000, 5000].iter() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        let lag = 2;
        let mu: f32 = 0.;
        let delta = 0.5;
        let sigma = 1.0;
        let y = gen_ar_1(&mut rng, *size, mu, delta, sigma);

        c.bench_with_input(BenchmarkId::new("adf_f32_2", size), &y, |b, y| {
            b.iter(|| adf_test(y, lag, Regression::Constant))
        });
    }
}

fn adf_benchmark_f64_2(c: &mut Criterion) {
    for size in [100, 200, 500, 1000, 5000].iter() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        let lag = 2;
        let mu: f64 = 0.;
        let delta = 0.5;
        let sigma = 1.0;
        let y = gen_ar_1(&mut rng, *size, mu, delta, sigma);

        c.bench_with_input(BenchmarkId::new("adf_f64_2", size), &y, |b, y| {
            b.iter(|| adf_test(y, lag, Regression::Constant))
        });
    }
}

fn adf_benchmark_f32_10(c: &mut Criterion) {
    for size in [100, 200, 500, 1000, 5000].iter() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        let lag = 10;
        let mu: f32 = 0.;
        let delta = 0.5;
        let sigma = 1.0;
        let y = gen_ar_1(&mut rng, *size, mu, delta, sigma);

        c.bench_with_input(BenchmarkId::new("adf_f32_10", size), &y, |b, y| {
            b.iter(|| adf_test(y, lag, Regression::Constant))
        });
    }
}

fn adf_benchmark_f64_10(c: &mut Criterion) {
    for size in [100, 200, 500, 1000, 5000].iter() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        let lag = 10;
        let mu: f64 = 0.;
        let delta = 0.5;
        let sigma = 1.0;
        let y = gen_ar_1(&mut rng, *size, mu, delta, sigma);

        c.bench_with_input(BenchmarkId::new("adf_f64_10", size), &y, |b, y| {
            b.iter(|| adf_test(y, lag, Regression::Constant))
        });
    }
}

criterion_group!(
    benches,
    adf_benchmark_f32_2,
    adf_benchmark_f64_2,
    adf_benchmark_f32_10,
    adf_benchmark_f64_10
);
criterion_main!(benches);
