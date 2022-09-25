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

use super::AlphaLevel;

/// Approximate Dickey-Fuller distribution for specific alpha levels
/// for model 1 (constant, no trend): $Δy_i = β_0 + β_1*y_{i-1} + ε_i$
/// https://www.real-statistics.com/statistics-tables/augmented-dickey-fuller-table/
pub fn model_1_approx_critical_value(sz: usize, alpha: AlphaLevel) -> f64 {
    let (t, u, v, w) = match alpha {
        AlphaLevel::OnePercent => (-3.43035, -6.5393, -16.786, -79.433),
        AlphaLevel::TwoPointFivePercent => (-3.1175, -4.53235, -9.8824, -57.7669),
        AlphaLevel::FivePercent => (-2.86154, -2.86154, -4.234, -40.04),
        AlphaLevel::TenPercent => (-2.56677, -1.5384, -2.809, 0.),
    };

    t + u / sz as f64 + v / (sz as f64).powi(2) + w / (sz as f64).powi(3)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_model_1_critical_approx_value_25() {
        assert_relative_eq!(
            model_1_approx_critical_value(25, AlphaLevel::OnePercent),
            -3.724,
            epsilon = 1e-3
        );
        assert_relative_eq!(
            model_1_approx_critical_value(25, AlphaLevel::TwoPointFivePercent),
            -3.318,
            epsilon = 1e-3
        );
        assert_relative_eq!(
            model_1_approx_critical_value(25, AlphaLevel::FivePercent),
            -2.986,
            epsilon = 1e-3
        );
        assert_relative_eq!(
            model_1_approx_critical_value(25, AlphaLevel::TenPercent),
            -2.633,
            epsilon = 1e-3
        );
    }

    #[test]
    fn test_model_1_critical_approx_value_100() {
        assert_relative_eq!(
            model_1_approx_critical_value(100, AlphaLevel::OnePercent),
            -3.498,
            epsilon = 1e-3
        );
        assert_relative_eq!(
            model_1_approx_critical_value(100, AlphaLevel::TwoPointFivePercent),
            -3.164,
            epsilon = 1e-3
        );
        assert_relative_eq!(
            model_1_approx_critical_value(100, AlphaLevel::FivePercent),
            -2.891,
            epsilon = 1e-3
        );
        assert_relative_eq!(
            model_1_approx_critical_value(100, AlphaLevel::TenPercent),
            -2.582,
            epsilon = 1e-3
        );
    }

    #[test]
    fn test_model_1_critical_approx_value_500() {
        assert_relative_eq!(
            model_1_approx_critical_value(500, AlphaLevel::OnePercent),
            -3.443,
            epsilon = 1e-3
        );
        assert_relative_eq!(
            model_1_approx_critical_value(500, AlphaLevel::TwoPointFivePercent),
            -3.127,
            epsilon = 1e-3
        );
        assert_relative_eq!(
            model_1_approx_critical_value(500, AlphaLevel::FivePercent),
            -2.867,
            epsilon = 1e-3
        );
        assert_relative_eq!(
            model_1_approx_critical_value(500, AlphaLevel::TenPercent),
            -2.570,
            epsilon = 1e-3
        );
    }
}
