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

//! Example of the Augmented Dickey-Fuller test
use unit_root::prelude::distrib::{AlphaLevel, Regression};
use unit_root::prelude::nalgebra::DVector;
use unit_root::prelude::*;

fn main() {
    let y = DVector::from_row_slice(&[
        -0.89642362f64,
        0.3222552,
        -1.96581989,
        -1.10012936,
        -1.3682928,
        1.17239875,
        2.19561259,
        2.54295031,
        2.05530587,
        1.13212955,
        -0.42968979,
    ]);

    // compute the test statistic
    let lag = 1;
    let regression = Regression::Constant;
    let report = tools::adf_test(&y, lag, regression).unwrap();

    // critical values for the model with a constant but no trend:
    let critical_value =
        distrib::dickeyfuller::get_critical_value(regression, report.size, AlphaLevel::OnePercent)
            .unwrap();
    assert_eq!(report.size, 9);

    // comparison
    let t_stat = report.test_statistic;
    println!("t-statistic: {}", t_stat);
    assert!((t_stat - -1.1639935).abs() < 1e-6);
    assert!(t_stat > critical_value);
}
