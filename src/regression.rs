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

use nalgebra::{DMatrix, DVector, RealField, Scalar};
use num_traits::Float;

use crate::prelude::Error;

/// Returns the beta coefficients and t-statistics of the OLS regression of y on x.
/// Note: the intercept is the first coefficient.
pub fn ols<F: Float + Scalar + RealField>(
    y: &DVector<F>,
    x: &DMatrix<F>,
) -> Result<(DVector<F>, DVector<F>), Error> {
    // Augment X with a column of 1s for the intercept - in first column
    let a = x.clone();
    let a = a.insert_column(0, F::from(1.0).unwrap());

    let n = x.nrows();
    let k = a.ncols();

    let at = &a.transpose();
    // beta = (A'A)^-1 A'y
    let ata = at * &a;
    let ata_inv = &ata
        .try_inverse()
        .ok_or_else(|| Error::FailedToInvertMatrix("OLS failed to invert A.T*A".into()))?;
    let aty = at * y;

    // the regression coefficients
    let beta_ = ata_inv * aty;

    // the predicted values
    let y_hat = a * &beta_;

    // the residuals
    let residuals = y - y_hat;

    let rtr = &residuals.transpose() * &residuals;
    let rtr = rtr.get((0, 0)).unwrap();

    // The variance of the residuals
    let vcv = ata_inv * (*rtr / F::from(n - k).unwrap());

    // The standard errors of the coefficients
    let se = vcv.diagonal().map(|x| Float::sqrt(x));

    let t_statistics = beta_.component_div(&se);

    Ok((beta_, t_statistics))
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::{DMatrix, DVector};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use crate::utils::{gen_affine_data, gen_affine_data_with_whitenoise};

    #[test]
    fn test_ols_f32() {
        let y = DVector::from_row_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0]);
        let x = DMatrix::from_row_slice(5, 1, &[1.0f32, 2.0, 3.0, 4.0, 5.0]);

        let (beta_hat, t_stats) = super::ols(&y, &x).unwrap();

        assert_eq!(beta_hat.get(0).unwrap().to_owned(), 0.0);
        assert_eq!(beta_hat.get(1).unwrap().to_owned(), 1.0);
        assert!(t_stats.get(0).unwrap().is_nan());
        assert!(t_stats.get(1).unwrap().is_infinite());
    }

    #[test]
    fn test_ols_f64() {
        let y = DVector::from_row_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0]);
        let x = DMatrix::from_row_slice(5, 1, &[1.0f64, 2.0, 3.0, 4.0, 5.0]);

        let (beta_hat, t_stats) = super::ols(&y, &x).unwrap();

        assert_eq!(beta_hat.get(0).unwrap().to_owned(), 0.0);
        assert_eq!(beta_hat.get(1).unwrap().to_owned(), 1.0);
        assert!(t_stats.get(0).unwrap().is_nan());
        assert!(t_stats.get(1).unwrap().is_infinite());
    }

    #[test]
    fn test_ols_2() {
        let sz = 400;

        let mu = 4.0;
        let beta = 12.;

        let (x, y) = gen_affine_data(sz, mu, beta);

        let (beta_hat, t_stats) = super::ols(&y, &x).unwrap();
        let mu_hat = beta_hat.get(0).unwrap().to_owned();
        let beta_hat = beta_hat.get(1).unwrap().to_owned();

        assert_relative_eq!(mu_hat, mu, epsilon = 0.1);
        assert_relative_eq!(beta_hat, beta, epsilon = 0.1);

        let t_stat_mu = t_stats.get(0).unwrap().to_owned();
        let t_stat_beta = t_stats.get(1).unwrap().to_owned();

        assert!(t_stat_mu > 1e3);
        assert!(t_stat_beta > 1e3);
    }

    #[test]
    fn test_ols_with_gaussian_noise() {
        let sz = 400;

        let mu = 43.0;
        let beta = 2.;

        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let (x, y) = gen_affine_data_with_whitenoise(&mut rng, sz, mu, beta);

        let (beta_hat, t_stats) = super::ols(&y, &x).unwrap();
        let mu_hat = beta_hat.get(0).unwrap().to_owned();
        let beta_hat = beta_hat.get(1).unwrap().to_owned();

        // println!("mu_hat: {}", mu_hat);
        // println!("beta_hat: {}", beta_hat);

        assert_relative_eq!(mu_hat, mu, epsilon = 2.);
        assert_relative_eq!(beta_hat, beta, epsilon = 0.5);

        let t_stat_mu = t_stats.get(0).unwrap().to_owned();
        let t_stat_beta = t_stats.get(1).unwrap().to_owned();

        assert!(t_stat_mu > 100.);
        assert!(t_stat_beta > 100.);
    }
}
