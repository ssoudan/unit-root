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

//! Basic Unit root tests for time series data.
//!
//! # Examples
//!
//! ```rust
//! use unit_root::prelude::distrib::dickeyfuller::get_critical_value;
//! use unit_root::prelude::distrib::{AlphaLevel, Regression};
//! use unit_root::prelude::nalgebra::DVector;
//! use unit_root::prelude::tools::adf_test;
//! use unit_root::prelude::*;
//!
//! let y = DVector::from_row_slice(&[
//!     -0.89642362f64,
//!     0.3222552,
//!     -1.96581989,
//!     -1.10012936,
//!     -1.3682928,
//!     1.17239875,
//!     2.19561259,
//!     2.54295031,
//!     2.05530587,
//!     1.13212955,
//!     -0.42968979,
//! ]);
//!
//! let lag = 2;
//! let regression = Regression::Constant;
//! let report = adf_test(&y, lag, regression).unwrap();
//!
//! let critical_value: f64 =
//!     get_critical_value(regression, report.size, AlphaLevel::OnePercent).unwrap();
//! assert_eq!(report.size, 8);
//!
//! let t_stat = report.test_statistic;
//! println!("t-statistic: {}", t_stat);
//! println!("critical_value: {}", critical_value);
//! ```
//!
//! # References
//! - [Augmented Dickey-Fuller test](https://en.wikipedia.org/wiki/Augmented_Dickey–Fuller_test)
//! - [Dickey-Fuller test](https://en.wikipedia.org/wiki/Dickey%E2%80%93Fuller_test)
//! - [Statsmodels](https://github.com/statsmodels/statsmodels/blob/main/statsmodels/tsa/stattools.py)
//! - [Dickey-Fuller Test](https://www.real-statistics.com/time-series-analysis/stochastic-processes/dickey-fuller-test/)
//! - [Augmented Dickey-Fuller Test](https://www.real-statistics.com/time-series-analysis/stochastic-processes/augmented-dickey-fuller-test/)
//! - [Augmented Dickey-Fuller Table](https://www.real-statistics.com/statistics-tables/augmented-dickey-fuller-table/)
//! - [Standard errors in OLS](https://lukesonnet.com/teaching/inference/200d_standard_errors.pdf)
use thiserror::Error;

pub(crate) mod distrib;
pub(crate) mod tools;

/// The public API.
pub mod prelude;

#[cfg(any(feature = "unstable", test))]
/// unstable utils API
pub mod utils;

#[cfg(any(feature = "unstable", test))]
/// unstable regression API
pub mod regression;

#[cfg(not(any(feature = "unstable", test)))]
pub(crate) mod regression;

/// The error type for this crate.
#[derive(Debug, Clone, Error)]
pub enum Error {
    /// Failed to invert matrix.
    #[error("Failed to invert matrix: {0}")]
    FailedToInvertMatrix(String),
    /// NotEnoughSamples
    #[error("Not enough samples")]
    NotEnoughSamples,
    /// Failed to convert float.
    #[error("Failed to convert float")]
    ConversionFailed,
}
