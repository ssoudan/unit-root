# Unit root tests in Rust

![Build](https://github.com/ssoudan/unit-root/actions/workflows/rust.yml/badge.svg)

## Description

Stationarity tests for time-series data in Rust. 

At the moment:
[Dickey-Fuller test](https://en.wikipedia.org/wiki/Dickey–Fuller_test) and 
[Augmented Dickey-Fuller test](https://en.wikipedia.org/wiki/Augmented_Dickey–Fuller_test) with a 
constant but no trend.  

## License 

This project is licensed under the terms of the Apache License 2.0.

## Usage

Augmented Dickey-Fuller test:

```rust
use unit_root::prelude::distrib::{AlphaLevel,Regression};
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

    let lag = 2;
    
    // compute the test statistic
    let regression = Regression::Constant;
    let report = tools::adf::adf_test(&y, lag, regression).unwrap();

    // critical values for the model with a constant but no trend:
    let critical_value = distrib::dickeyfuller::get_critical_value(
        regression,
        report.size,
        AlphaLevel::OnePercent,
    )
    .unwrap();
    assert_eq!(report.size, 10);

    // comparison
    let t_stat = report.test_statistic;
    println!("t-statistic: {}", t_stat);
    println!("critical value: {}", critical_value);
}
```

See [examples](examples/) for more.