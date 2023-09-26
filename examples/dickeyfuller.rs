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

//! A simple example of how to use the library
use unit_root::prelude::distrib::dickeyfuller::constant_no_trend_critical_value;
use unit_root::prelude::distrib::AlphaLevel;
use unit_root::prelude::nalgebra::DVector;
use unit_root::prelude::*;

fn main() {
    let y = DVector::from_row_slice(&[
        -0.89642362,
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

    let report = tools::dickeyfuller::constant_no_trend_test(&y).unwrap();

    let critical_value = constant_no_trend_critical_value(report.size, AlphaLevel::OnePercent).unwrap();
    assert_eq!(report.size, 10);

    let t_stat = report.test_statistic;
    println!("t-statistic: {}", t_stat);
    assert!((t_stat - -1.472691f64).abs() < 1e-6);
    assert!(t_stat > critical_value);
    // cannot reject the hypothesis that the series is not stationary
}
