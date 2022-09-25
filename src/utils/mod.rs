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

//! Utilities
use nalgebra::{DMatrix, DVector};
use rand::prelude::Distribution;
use rand::Rng;
use rand_distr::StandardNormal;

/// Generates AR(1) data:
/// Y_t = mu + delta * Y_{t-1} + sigma * e_t
/// where e_t is a standard normal random variable
pub fn gen_ar_1<R: Rng + ?Sized>(
    mut rng: &mut R,
    size: usize,
    mu: f64,
    delta: f64,
    sigma: f64,
) -> DVector<f64> {
    let mut y = DVector::zeros(size);

    let epsilon: f64 = StandardNormal.sample(&mut rng);
    y[0] = mu + delta * 0.0 + sigma * epsilon;

    for i in 1..size {
        let epsilon: f64 = StandardNormal.sample(&mut rng);
        y[i] = mu + delta * y[i - 1] + sigma * epsilon;
    }

    y
}

fn gen_x(sz: usize) -> DMatrix<f64> {
    DMatrix::from_row_slice(
        sz,
        1,
        Vec::from_iter(0..sz)
            .into_iter()
            .map(|x| x as f64)
            .collect::<Vec<f64>>()
            .as_slice(),
    )
}

/// Generate data as y = beta * x + mu
/// where noise is drawn from a standard normal distribution
/// Returns (x, y).
pub fn gen_affine_data(sz: usize, mu: f64, beta: f64) -> (DMatrix<f64>, DVector<f64>) {
    let x = gen_x(sz);
    let y = (beta * &x).add_scalar(mu);

    let y = DVector::from_row_slice(y.as_slice());
    (x, y)
}

/// Generate data as y = beta * x + mu + noise
/// where noise is drawn from a standard normal distribution
/// Returns (x, y).
pub fn gen_affine_data_with_whitenoise<R: Rng + ?Sized>(
    mut rng: &mut R,
    sz: usize,
    mu: f64,
    beta: f64,
) -> (DMatrix<f64>, DVector<f64>) {
    let x = gen_x(sz);
    let y = beta * x.clone();

    let noise = DVector::from_iterator(sz, StandardNormal.sample_iter(&mut rng).take(sz));
    let y = (y + noise).add_scalar(mu);
    (x, y)
}
