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

use super::{AlphaLevel, CalculationError, Regression};

/// Approximate Dickey-Fuller distribution for specific alpha levels
/// for constant, no trend: $Δy_i = β_0 + β_1*y_{i-1} + ε_i$
/// https://www.real-statistics.com/statistics-tables/augmented-dickey-fuller-table/
pub fn constant_no_trend_critical_value<F: Float>(
    sz: usize,
    alpha: AlphaLevel,
) -> Result<F, CalculationError> {
    let (t, u, v, w) = match alpha {
        AlphaLevel::OnePercent => (-3.43035, -6.5393, -16.786, -79.433),
        AlphaLevel::TwoPointFivePercent => (-3.1175, -4.53235, -9.8824, -57.7669),
        AlphaLevel::FivePercent => (-2.86154, -2.86154, -4.234, -40.04),
        AlphaLevel::TenPercent => (-2.56677, -1.5384, -2.809, 0.),
    };
    return calculate_t_stat_from_estimators(t, u, v, w, sz);
}

pub fn no_constant_no_trend_critical_value<F: Float>(
    sz: usize,
    alpha: AlphaLevel,
) -> Result<F, CalculationError> {
    let (t, u, v, w) = match alpha {
        AlphaLevel::OnePercent => (-2.56574, -2.2358, -3.627, 0.),
        AlphaLevel::TwoPointFivePercent => (-2.222133, -1.15384, -3.4829, 17.17265),
        AlphaLevel::FivePercent => (-1.941, -0.2686, -3.365, 31.223),
        AlphaLevel::TenPercent => (-1.61682, 0.2656, -2.714, 25.364),
    };

    return calculate_t_stat_from_estimators(t, u, v, w, sz);
}

pub fn constant_trend_critical_value<F: Float>(
    sz: usize,
    alpha: AlphaLevel,
) -> Result<F, CalculationError> {
    let (t, u, v, w) = match alpha {
        AlphaLevel::OnePercent => (-3.95877, -9.0531, -28.428, -134.155),
        AlphaLevel::TwoPointFivePercent => (-3.657216, -6.488615, -17.7624, -85.32545),
        AlphaLevel::FivePercent => (-3.41049, -4.3904, -9.036, -45.374),
        AlphaLevel::TenPercent => (-3.12705, -2.5856, -3.925, -22.38),
    };

    return calculate_t_stat_from_estimators(t, u, v, w, sz);
}

pub fn get_critical_value<F: Float>(
    regression: Regression,
    sz: usize,
    alpha: AlphaLevel,
) -> Result<F, CalculationError> {
    return match regression {
        Regression::Constant => constant_no_trend_critical_value(sz, alpha),
        Regression::ConstantAndTrend => constant_trend_critical_value(sz, alpha),
        Regression::NoConstantNoTrend => no_constant_no_trend_critical_value(sz, alpha),
    };
}

fn calculate_t_stat_from_estimators<F: Float>(
    t: f64,
    u: f64,
    v: f64,
    w: f64,
    sz: usize,
) -> Result<F, CalculationError> {
    let n = sz as f64;
    let t_stat = t + (u / n) + (v / n.powi(2)) + (w / n.powi(3));
    let x = F::from(t_stat).ok_or(CalculationError::ConversionFailed)?;
    Ok(x)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_model_1_critical_approx_value_25() {
        assert_relative_eq!(
            constant_no_trend_critical_value::<f32>(25, AlphaLevel::OnePercent)
                .expect("failed to convert float"),
            -3.724,
            epsilon = 1e-3
        );
        assert_relative_eq!(
            constant_no_trend_critical_value::<f32>(25, AlphaLevel::TwoPointFivePercent)
                .expect("failed to convert float"),
            -3.318,
            epsilon = 1e-3
        );
        assert_relative_eq!(
            constant_no_trend_critical_value::<f32>(25, AlphaLevel::FivePercent)
                .expect("failed to convert float"),
            -2.986,
            epsilon = 1e-3
        );
        assert_relative_eq!(
            constant_no_trend_critical_value::<f32>(25, AlphaLevel::TenPercent)
                .expect("failed to convert float"),
            -2.633,
            epsilon = 1e-3
        );
    }

    #[test]
    fn test_model_1_critical_approx_value_100() {
        assert_relative_eq!(
            constant_no_trend_critical_value::<f32>(100, AlphaLevel::OnePercent)
                .expect("failed to convert float"),
            -3.498,
            epsilon = 1e-3
        );
        assert_relative_eq!(
            constant_no_trend_critical_value::<f32>(100, AlphaLevel::TwoPointFivePercent)
                .expect("failed to convert float"),
            -3.164,
            epsilon = 1e-3
        );
        assert_relative_eq!(
            constant_no_trend_critical_value::<f32>(100, AlphaLevel::FivePercent)
                .expect("failed to convert float"),
            -2.891,
            epsilon = 1e-3
        );
        assert_relative_eq!(
            constant_no_trend_critical_value::<f32>(100, AlphaLevel::TenPercent)
                .expect("failed to convert float"),
            -2.582,
            epsilon = 1e-3
        );
    }

    #[test]
    fn test_model_1_critical_approx_value_500() {
        assert_relative_eq!(
            constant_no_trend_critical_value::<f32>(500, AlphaLevel::OnePercent)
                .expect("failed to convert float"),
            -3.443,
            epsilon = 1e-3
        );
        assert_relative_eq!(
            constant_no_trend_critical_value::<f32>(500, AlphaLevel::TwoPointFivePercent)
                .expect("failed to convert float"),
            -3.127,
            epsilon = 1e-3
        );
        assert_relative_eq!(
            constant_no_trend_critical_value::<f32>(500, AlphaLevel::FivePercent)
                .expect("failed to convert float"),
            -2.867,
            epsilon = 1e-3
        );
        assert_relative_eq!(
            constant_no_trend_critical_value::<f32>(500, AlphaLevel::TenPercent)
                .expect("failed to convert float"),
            -2.570,
            epsilon = 1e-3
        );
    }

    #[test]
    fn test_critical_approx_value_no_constant_no_trend() {
        let epsilon = 1e-3;

        let test_data = [
            (AlphaLevel::OnePercent, -2.661, 25),
            (AlphaLevel::TwoPointFivePercent, -2.273, 25),
            (AlphaLevel::FivePercent, -1.955, 25),
            (AlphaLevel::TenPercent, -1.609, 25),
        ];
        for (alpha, expected_value, sz) in test_data {
            assert_relative_eq!(
                no_constant_no_trend_critical_value::<f32>(sz, alpha)
                    .expect("failed to convert float"),
                expected_value,
                epsilon = epsilon
            );
        }
    }

    #[test]
    fn test_critical_approx_value_constant_trend() {
        let epsilon = 1e-3;

        let test_data = [
            (AlphaLevel::OnePercent, -4.375, 25),
            (AlphaLevel::TwoPointFivePercent, -3.951, 25),
            (AlphaLevel::FivePercent, -3.603, 25),
            (AlphaLevel::TenPercent, -3.238, 25),
        ];
        for (alpha, expected_value, sz) in test_data {
            assert_relative_eq!(
                constant_trend_critical_value::<f32>(sz, alpha).expect("failed to convert float"),
                expected_value,
                epsilon = epsilon
            );
        }
    }

    #[test]
    fn test_get_critical_value() {
        let epsilon = 1e-3;
        let test_data = [
            (
                Regression::NoConstantNoTrend,
                100,
                AlphaLevel::TwoPointFivePercent,
                -2.234,
            ),
            (Regression::Constant, 25, AlphaLevel::OnePercent, -3.724),
            (
                Regression::ConstantAndTrend,
                50,
                AlphaLevel::TenPercent,
                -3.181,
            ),
        ];
        for (regression, sz, alpha, expected_value) in test_data {
            assert_relative_eq!(
                get_critical_value::<f32>(regression, sz, alpha).expect("failed to convert float"),
                expected_value,
                epsilon = epsilon
            );
        }
    }
}
