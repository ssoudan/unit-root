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

//! Augmented Dickey-Fuller test
use nalgebra::{DVector, RealField, Scalar};
use num_traits::Float;

use crate::distrib::Regression;
use crate::prelude::tools::Report;
use crate::regression::ols;
use crate::{tools, Error};

/// Augmented Dickey-Fuller test
/// - Constant and no trend model
/// - Fixed lag
/// - y must have strictly more than n + 1 elements.
pub fn constant_no_trend_test<F: RealField + Scalar + Float>(
    y: &DVector<F>,
    lag: usize,
) -> Result<Report<F>, Error> {
    let (delta_y, x, size) = tools::prepare(y, lag, Regression::Constant)?;

    let (_betas, t_stats) = ols(&delta_y, &x)?;

    Ok(Report {
        test_statistic: t_stats[1],
        size,
    })
}

#[cfg(test)]
mod tests {
    use nalgebra::DVector;

    use crate::prelude::tools::dickeyfuller;

    #[test]
    fn test_t_statistics() {
        let lag = 2;
        let y = DVector::from_row_slice(&[
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
        ]);

        let report = super::constant_no_trend_test(&y, lag).unwrap();
        assert_eq!(report.size, 8);
        assert_eq!(report.test_statistic, 0.48612142266202985f64);
        // statsmodels.tsa.stattools.adfuller(y, maxlag=2) gives:
        // ADF Statistic: 0.486121
        // p-value: 0.984445
        // Critical Values:
        // 1%: -4.665
        // 5%: -3.367
        // 10%: -2.803
        // usedlag= 2
    }

    #[test]
    fn test_adf_lag_0_is_dickeyfuller_test() {
        let lag = 0;

        let y = vec![1_f32, 3., 6., 10., 15., 21., 28., 36., 45., 55.];

        let y = DVector::from(y);

        let report = super::constant_no_trend_test(&y, lag).unwrap();
        let df_report = dickeyfuller::constant_no_trend_test(&y).unwrap();

        assert_eq!(report.test_statistic, df_report.test_statistic);
        assert_eq!(report.size, df_report.size);
    }
}
