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

use nalgebra::{DMatrix, RealField, Scalar};
use num_traits::Float;

use crate::prelude::nalgebra::DVector;
use crate::regression::ols;

/// Univariate Dickey-Fuller test report
pub struct Report<F> {
    /// The test statistic - when available
    pub test_statistic: Option<F>,
    /// The size of the sample
    pub size: usize,
}

/// returns the t-statistic of the Dickey-Fuller test
/// and the size of the sample.
///
/// The null hypothesis is that the series is non-stationary.
///
/// # Details
///
/// Critical values for **model 1**  (constant, no trend): $\Delta y_t = \mu +
/// \delta*y_{t-1} + \epsilon_i $ can obtained
/// from `unit_root::prelude::distrib::dickeyfuller::model_1_approx_critical_value`.
///
/// - If $t_{stat} < \mathrm{t_{\mathrm{crit}}(\alpha)}$ then reject $H_0$ at
/// $alpha$ significance level - and thus conclude that the series is stationary.
/// - If $t_{stat} > \mathrm{t_{\mathrm{crit}}(\alpha)}$ then fail to reject $H_0$ at
/// $alpha$ significance level - and thus conclude we cannot reject the hypothesis that
/// the series is not stationary.
///
/// # Examples:
///
/// ```rust
/// use unit_root::prelude::distrib::dickeyfuller::constant_no_trend_critical_value;
/// use unit_root::prelude::distrib::AlphaLevel;
/// use unit_root::prelude::nalgebra::DVector;
/// use unit_root::prelude::*;
///
/// let y = DVector::from_row_slice(&[
///     -0.89642362,
///     0.3222552,
///     -1.96581989,
///     -1.10012936,
///     -1.3682928,
///     1.17239875,
///     2.19561259,
///     2.54295031,
///     2.05530587,
///     1.13212955,
///     -0.42968979,
/// ]);
///
/// let report = tools::dickeyfuller::constant_no_trend_test(&y);
///
/// let critical_value = constant_no_trend_critical_value(report.size, AlphaLevel::OnePercent);
/// assert_eq!(report.size, 10);
///
/// let t_stat = report.test_statistic.unwrap();
/// println!("t-statistic: {}", t_stat);
/// assert!((t_stat - -1.472691f64).abs() < 1e-6);
/// assert!(t_stat > critical_value);
/// ```
pub fn constant_no_trend_test<F: Float + Scalar + RealField>(series: &DVector<F>) -> Report<F> {
    let (delta_y, y_t_1, size) = diff(series);

    let (_betas, t_stats) = ols(&delta_y, &y_t_1).unwrap();

    let t_stat_beta_1 = t_stats[1];

    if t_stat_beta_1.is_finite() {
        Report {
            test_statistic: Some(t_stat_beta_1),
            size,
        }
    } else {
        Report {
            test_statistic: None,
            size,
        }
    }
}

/// returns Delta[y(t)] and y(t-1)
fn diff<F: Scalar + RealField>(y: &DVector<F>) -> (DVector<F>, DMatrix<F>, usize) {
    let n = y.len();

    let y_t_1 = y.clone().remove_row(n - 1);
    let y_t = y.clone().remove_row(0);
    let delta = y_t - &y_t_1;

    let y_t_1: DMatrix<F> = DMatrix::from_row_slice(n - 1, 1, y_t_1.as_slice());

    (delta, y_t_1, n - 1)
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::distrib::dickeyfuller::constant_no_trend_critical_value;
    use crate::distrib::AlphaLevel;
    use crate::utils::gen_ar_1;

    #[test]
    fn test_dickeyfuller_no_unit_root_f32() {
        let n = 100;

        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let delta: f32 = 0.5;
        let y = gen_ar_1(&mut rng, n, 0.0, delta, 1.0);

        println!("y={}", y);
        let report = constant_no_trend_test(&y);

        let critical_value = constant_no_trend_critical_value(report.size, AlphaLevel::OnePercent);

        assert!(report.test_statistic.is_some());
        let t_stat = report.test_statistic.unwrap();
        assert!(t_stat < critical_value);
    }

    #[test]
    fn test_dickeyfuller_with_unit_root_f32() {
        let n = 100;

        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let delta: f32 = 1.0;
        let y = gen_ar_1(&mut rng, n, 0.0, delta, 1.0);

        println!("y={}", y);
        let report = constant_no_trend_test(&y);

        let critical_value = constant_no_trend_critical_value(report.size, AlphaLevel::OnePercent);

        assert!(report.test_statistic.is_some());
        let t_stat = report.test_statistic.unwrap();
        assert!(t_stat > critical_value);
    }

    #[test]
    fn test_dickeyfuller_no_unit_root_f64() {
        let n = 100;

        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let delta: f64 = 0.5;
        let y = gen_ar_1(&mut rng, n, 0.0, delta, 1.0);

        println!("y={}", y);
        let report = constant_no_trend_test(&y);

        let critical_value = constant_no_trend_critical_value(report.size, AlphaLevel::OnePercent);

        assert!(report.test_statistic.is_some());
        let t_stat = report.test_statistic.unwrap();
        assert!(t_stat < critical_value);
    }

    #[test]
    fn test_dickeyfuller_with_unit_root_f64() {
        let n = 100;

        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let delta: f64 = 1.0;
        let y = gen_ar_1(&mut rng, n, 0.0, delta, 1.0);

        println!("y={}", y);
        let report = constant_no_trend_test(&y);

        let critical_value = constant_no_trend_critical_value(report.size, AlphaLevel::OnePercent);

        assert!(report.test_statistic.is_some());
        let t_stat = report.test_statistic.unwrap();
        assert!(t_stat > critical_value);
    }
}
