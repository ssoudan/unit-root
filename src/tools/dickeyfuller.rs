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

use nalgebra::{RealField, Scalar};
use num_traits::Float;

use crate::distrib::Regression;
use crate::prelude::nalgebra::DVector;
use crate::prelude::tools::Report;
use crate::regression::ols;
use crate::tools::prepare;
use crate::Error;

/// Returns the t-statistic of the Dickey-Fuller test
/// and the size of the sample.
///
/// The null hypothesis is that the series is non-stationary.
///
/// # Details
///
/// Critical values for can obtained from
/// `unit_root::prelude::distrib::dickeyfuller::get_critical_value`.
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
/// use unit_root::prelude::distrib::{AlphaLevel, Regression};
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
/// let regression = Regression::Constant;
///
/// let report = tools::dickeyfuller_test(&y, regression).unwrap();
///
/// let critical_value =
///     distrib::dickeyfuller::get_critical_value(regression, report.size, AlphaLevel::OnePercent)
///         .unwrap();
/// assert_eq!(report.size, 10);
///
/// let t_stat = report.test_statistic;
/// println!("t-statistic: {}", t_stat);
/// assert!((t_stat - -1.472691f64).abs() < 1e-6);
/// assert!(t_stat > critical_value);
/// ```
pub fn dickeyfuller_test<F: Float + Scalar + RealField>(
    series: &DVector<F>,
    regression: Regression,
) -> Result<Report<F>, Error> {
    let (delta_y, y_t_1, size) = prepare(series, 0, regression)?;

    let (_betas, t_stats) = ols(&delta_y, &y_t_1)?;

    Ok(Report {
        test_statistic: t_stats[0],
        size,
    })
}

/// Comparison with statsmodels.tsa.stattools.adfuller use the following code - see
/// [`tools::adf_test::test`] for the definition of the function:
/// ```python
/// adf_test(y, maxlag=0, regression='n')
/// adf_test(y, maxlag=0, regression='c')
/// adf_test(y, maxlag=0, regression='ct')
/// ```
/// Note: maxlag is set to 0.
#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::distrib::dickeyfuller::constant_no_trend_critical_value;
    use crate::distrib::AlphaLevel;
    use crate::utils::gen_ar_1;

    const Y: [f64; 11] = [
        -1.06714348,
        -1.14700339,
        0.79204106,
        -0.05845247,
        -0.67476754,
        -0.10396661,
        1.82059282,
        -0.51169443,
        2.07712365,
        1.85668086,
        2.56363688,
    ];

    #[test]
    fn test_t_statistics_n() {
        let y = DVector::from_row_slice(&Y[..]);

        let report = dickeyfuller_test(&y, Regression::NoConstantNoTrend).unwrap();
        assert_eq!(report.size, 10);
        assert_relative_eq!(report.test_statistic, -1.5140129055f64, epsilon = 1e-9);
        // Results of Dickey-Fuller Test:
        // Test Statistic                 -1.5140129055
        // p-value                       0.121977783883
        // #Lags Used                                 0
        // Number of Observations Used               10
        // Critical Value (1%)                 -2.82559
        // Critical Value (5%)                -1.970287
        // Critical Value (10%)               -1.592036
    }

    #[test]
    fn test_t_statistics_c() {
        let y = DVector::from_row_slice(&Y[..]);

        let report = dickeyfuller_test(&y, Regression::Constant).unwrap();
        assert_eq!(report.size, 10);
        assert_relative_eq!(report.test_statistic, -1.83288396527f64, epsilon = 1e-9);
        // Results of Dickey-Fuller Test:
        // Test Statistic                -1.83288396527
        // p-value                       0.364262207135
        // #Lags Used                                 0
        // Number of Observations Used               10
        // Critical Value (1%)                -4.331573
        // Critical Value (5%)                 -3.23295
        // Critical Value (10%)                 -2.7487
    }

    #[test]
    fn test_t_statistics_ct() {
        let y = DVector::from_row_slice(&Y[..]);

        let report = dickeyfuller_test(&y, Regression::ConstantAndTrend).unwrap();
        assert_eq!(report.size, 10);
        assert_relative_eq!(report.test_statistic, -4.20337098854f64, epsilon = 1e-9);
        // Results of Dickey-Fuller Test:
        // Test Statistic                  -4.20337098854
        // p-value                       0.00442477220907
        // #Lags Used                                   0
        // Number of Observations Used                 10
        // Critical Value (1%)                  -5.282515
        // Critical Value (5%)                  -3.985264
        // Critical Value (10%)                  -3.44724
    }

    #[test]
    fn test_dickeyfuller_no_unit_root_f32() {
        let n = 100;

        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let delta: f32 = 0.5;
        let y = gen_ar_1(&mut rng, n, 0.0, delta, 1.0);

        let report = dickeyfuller_test(&y, Regression::Constant).unwrap();

        let critical_value =
            match constant_no_trend_critical_value(report.size, AlphaLevel::OnePercent) {
                Ok(v) => v,
                Err(_) => f32::MIN,
            };

        let t_stat = report.test_statistic;
        assert!(t_stat < critical_value);
    }

    #[test]
    fn test_dickeyfuller_with_unit_root_f32() {
        let n = 100;

        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let delta: f32 = 1.0;
        let y = gen_ar_1(&mut rng, n, 0.0, delta, 1.0);

        let report = dickeyfuller_test(&y, Regression::Constant).unwrap();

        let critical_value =
            match constant_no_trend_critical_value(report.size, AlphaLevel::OnePercent) {
                Ok(v) => v,
                Err(_) => f32::MAX,
            };

        let t_stat = report.test_statistic;
        assert!(t_stat > critical_value);
    }

    #[test]
    fn test_dickeyfuller_no_unit_root_f64() {
        let n = 100;

        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let delta: f64 = 0.5;
        let y = gen_ar_1(&mut rng, n, 0.0, delta, 1.0);

        let report = dickeyfuller_test(&y, Regression::Constant).unwrap();

        let critical_value =
            match constant_no_trend_critical_value(report.size, AlphaLevel::OnePercent) {
                Ok(v) => v,
                Err(_) => f64::MIN,
            };

        let t_stat = report.test_statistic;
        assert!(t_stat < critical_value);
    }

    #[test]
    fn test_dickeyfuller_with_unit_root_f64() {
        let n = 100;

        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let delta: f64 = 1.0;
        let y = gen_ar_1(&mut rng, n, 0.0, delta, 1.0);

        let report = dickeyfuller_test(&y, Regression::Constant).unwrap();

        let critical_value =
            match constant_no_trend_critical_value(report.size, AlphaLevel::OnePercent) {
                Ok(v) => v,
                Err(_) => f64::MAX,
            };

        let t_stat = report.test_statistic;
        assert!(t_stat > critical_value);
    }

    #[test]
    fn no_enough_data() {
        let y = DVector::from_row_slice(&[1.0]);
        let report = dickeyfuller_test(&y, Regression::Constant);
        assert!(report.is_err());
    }

    #[test]
    fn test_constant_no_trend_test() {
        let y = vec![1_f32, 3., 6., 10., 15., 21., 28., 36., 45., 55.];

        let y = DVector::from(y);

        let report = dickeyfuller_test(&y, Regression::Constant).unwrap();

        assert_eq!(report.size, 9);
    }
}
