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
