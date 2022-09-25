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

use num_traits::Float;

use super::AlphaLevel;

/// Approximate Dickey-Fuller distribution for specific alpha levels
/// for constant, no trend: $Δy_i = β_0 + β_1*y_{i-1} + ε_i$
/// https://www.real-statistics.com/statistics-tables/augmented-dickey-fuller-table/
pub fn constant_no_trend_critical_value<F: Float>(sz: usize, alpha: AlphaLevel) -> F {
    let (t, u, v, w) = match alpha {
        AlphaLevel::OnePercent => (-3.43035, -6.5393, -16.786, -79.433),
        AlphaLevel::TwoPointFivePercent => (-3.1175, -4.53235, -9.8824, -57.7669),
        AlphaLevel::FivePercent => (-2.86154, -2.86154, -4.234, -40.04),
        AlphaLevel::TenPercent => (-2.56677, -1.5384, -2.809, 0.),
    };

    let t = F::from(t).unwrap();
    let u = F::from(u).unwrap();
    let v = F::from(v).unwrap();
    let w = F::from(w).unwrap();

    let n = F::from(sz).unwrap();

    t + u / n + v / n.powi(2) + w / n.powi(3)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_model_1_critical_approx_value_25() {
        assert_relative_eq!(
            constant_no_trend_critical_value::<f32>(25, AlphaLevel::OnePercent),
            -3.724,
            epsilon = 1e-3
        );
        assert_relative_eq!(
            constant_no_trend_critical_value::<f32>(25, AlphaLevel::TwoPointFivePercent),
            -3.318,
            epsilon = 1e-3
        );
        assert_relative_eq!(
            constant_no_trend_critical_value::<f32>(25, AlphaLevel::FivePercent),
            -2.986,
            epsilon = 1e-3
        );
        assert_relative_eq!(
            constant_no_trend_critical_value::<f32>(25, AlphaLevel::TenPercent),
            -2.633,
            epsilon = 1e-3
        );
    }

    #[test]
    fn test_model_1_critical_approx_value_100() {
        assert_relative_eq!(
            constant_no_trend_critical_value::<f32>(100, AlphaLevel::OnePercent),
            -3.498,
            epsilon = 1e-3
        );
        assert_relative_eq!(
            constant_no_trend_critical_value::<f32>(100, AlphaLevel::TwoPointFivePercent),
            -3.164,
            epsilon = 1e-3
        );
        assert_relative_eq!(
            constant_no_trend_critical_value::<f32>(100, AlphaLevel::FivePercent),
            -2.891,
            epsilon = 1e-3
        );
        assert_relative_eq!(
            constant_no_trend_critical_value::<f32>(100, AlphaLevel::TenPercent),
            -2.582,
            epsilon = 1e-3
        );
    }

    #[test]
    fn test_model_1_critical_approx_value_500() {
        assert_relative_eq!(
            constant_no_trend_critical_value::<f32>(500, AlphaLevel::OnePercent),
            -3.443,
            epsilon = 1e-3
        );
        assert_relative_eq!(
            constant_no_trend_critical_value::<f32>(500, AlphaLevel::TwoPointFivePercent),
            -3.127,
            epsilon = 1e-3
        );
        assert_relative_eq!(
            constant_no_trend_critical_value::<f32>(500, AlphaLevel::FivePercent),
            -2.867,
            epsilon = 1e-3
        );
        assert_relative_eq!(
            constant_no_trend_critical_value::<f32>(500, AlphaLevel::TenPercent),
            -2.570,
            epsilon = 1e-3
        );
    }
}
