# Unit root tests in Rust

![Build](https://github.com/ssoudan/unit-root/actions/workflows/rust.yml/badge.svg)

Stationarity tests for time-series data. 
It's just Dickey-Fuller yet, but it's a start. 

## Usage

```rust
use unit_root::prelude::distrib::dickeyfuller::model_1_approx_critical_value;
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

    // compute the test statistic
    let report = tools::dickeyfuller::constant_no_trend_test(&y);
    
    // critical values for model 1 (constant but no trend):
    let critical_value = model_1_approx_critical_value(report.size, AlphaLevel::OnePercent);
    assert_eq!(report.size, 10);

    // comparison
    let t_stat = report.test_statistic.unwrap();
    println!("t-statistic: {}", t_stat);
    assert!((t_stat - -1.472691f32).abs() < 1e-6);
    assert!(t_stat > critical_value);
}
```