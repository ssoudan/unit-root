use std::fmt;

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
pub mod dickeyfuller;

/// Alpha levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaLevel {
    /// 1%
    OnePercent,
    /// 2.5%
    TwoPointFivePercent,
    /// 5%
    FivePercent,
    /// 10%
    TenPercent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Constant and trend parameters to include in regression.
pub enum Regression {
    /// constant only e.g.  Δy_i = β_0 + β_1*y_{i-1} + ε_i
    Constant,
    /// constant and trend e.g. Δy_i = β_0 + β_1*y_{i-1} + β_2*i + ε_i
    ConstantAndTrend,
    /// no constant, no trend e.g. Δy_i = β_1*y_{i-1}  + ε_i
    NoConstantNoTrend,
}

#[derive(Debug)]
pub enum CalculationError {
    ConversionFailed,
    // Other error variants...
}

impl std::fmt::Display for CalculationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::ConversionFailed => write!(f, "Conversion from f64 to generic float failed"),
            // Other error variants...
        }
    }
}

impl std::error::Error for CalculationError {}
