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
