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
pub fn adf_test<F: RealField + Scalar + Float>(
    y: &DVector<F>,
    lag: usize,
    regression: Regression,
) -> Result<Report<F>, Error> {
    let (delta_y, x, size) = tools::prepare(y, lag, regression)?;

    let (_betas, t_stats) = ols(&delta_y, &x)?;

    Ok(Report {
        test_statistic: t_stats[0],
        size,
    })
}

/// Comparison with statsmodels.tsa.stattools.adfuller use the following code:
/// ```python
/// import numpy as np
/// import pandas as pd
/// import statsmodels.tsa.stattools as ts
/// pd.options.display.float_format = '{:.12g}'.format
///
/// def adf_test(timeseries, maxlag=None, regression="c", autolag="AIC"):
///   print("Results of Dickey-Fuller Test:")
///   dftest = ts.adfuller(timeseries, maxlag=maxlag, regression=regression, autolag=autolag)
///   dfoutput = pd.Series(
///       dftest[0:4],
///       index=[
///           "Test Statistic",
///           "p-value",
///           "#Lags Used",
///           "Number of Observations Used",
///       ],
///   )
///   for key, value in dftest[4].items():
///       dfoutput["Critical Value (%s)" % key] = value
///   print(dfoutput)
///
/// y = [-1.06714348, -1.14700339,  0.79204106, -0.05845247, -0.67476754,
///      -0.10396661,  1.82059282, -0.51169443,  2.07712365,  1.85668086, 2.56363688]
///
/// adf_test(y, maxlag=2, regression='n')
/// adf_test(y, maxlag=2, regression='c')
/// adf_test(y, maxlag=2, regression='ct')
/// ```
///
/// Note: this library does not support the `autolag` yet. Tests are using the lag from
/// statsmodels.
#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::DVector;

    use crate::distrib::Regression;
    use crate::prelude::tools::{adf_test, dickeyfuller_test};

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
        let lag = 1;
        let y = DVector::from_row_slice(&Y[..]);

        let report = adf_test(&y, lag, Regression::NoConstantNoTrend).unwrap();
        assert_eq!(report.size, 9);
        assert_relative_eq!(report.test_statistic, -0.417100483298f64, epsilon = 1e-9);
        // Test Statistic                -0.417100483298
        // p-value                        0.529851882135
        // #Lags Used                                  1
        // Number of Observations Used                 9
        // Critical Value (1%)                  -2.85894
        // Critical Value (5%)            -1.96955775034
        // Critical Value (10%)           -1.58602219479
    }

    #[test]
    fn test_t_statistics_c() {
        let lag = 2;
        let y = DVector::from_row_slice(&Y[..]);

        let report = adf_test(&y, lag, Regression::Constant).unwrap();
        assert_eq!(report.size, 8);
        assert_relative_eq!(report.test_statistic, 0.486121422662f64, epsilon = 1e-9);
        // Results of Dickey-Fuller Test:
        // Test Statistic                0.486121422662
        // p-value                       0.984445107564
        // #Lags Used                                 2
        // Number of Observations Used                8
        // Critical Value (1%)           -4.66518632812
        // Critical Value (5%)             -3.367186875
        // Critical Value (10%)            -2.802960625
    }

    #[test]
    fn test_t_statistics_ct() {
        let lag = 0;
        let y = DVector::from_row_slice(&Y[..]);

        let report = adf_test(&y, lag, Regression::ConstantAndTrend).unwrap();
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
    fn test_adf_lag_0_is_dickeyfuller_test() {
        let lag = 0;

        let y = vec![1_f32, 3., 6., 10., 15., 21., 28., 36., 45., 55.];

        let y = DVector::from(y);

        let regression = Regression::Constant;

        let report = adf_test(&y, lag, regression).unwrap();
        let df_report = dickeyfuller_test(&y, regression).unwrap();

        assert_eq!(report.test_statistic, df_report.test_statistic);
        assert_eq!(report.size, df_report.size);
    }
}
