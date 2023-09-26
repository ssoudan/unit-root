use std::fmt::Debug;

use nalgebra::{DMatrix, DVector, RealField, Scalar};
use num_traits::Float;

use crate::{Error, regression, distrib::Regression};

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
pub(crate) mod adf;
pub(crate) mod dickeyfuller;

/// Test report
#[derive(Debug, Clone)]
pub struct Report<F: Debug + Clone> {
    /// The test statistic
    pub test_statistic: F,
    /// The size of the sample
    pub size: usize,
}

/// Returns Delta(y) = y - y.shift(1) and a matrix made of:
/// - a column of y.shift(1)
/// - n columns of Delta(y).shit(n)
pub(crate) fn prepare<F: RealField + Scalar + Float>(
    y: &DVector<F>,
    n: usize,
    regression: Regression
) -> Result<(DVector<F>, DMatrix<F>, usize), Error> {
    let y_len = y.len();

    if y_len <= n + 1 {
        return Err(Error::NotEnoughSamples);
    }

    // remove last element to build y[t-1]
    // it's length is y_len - 1
    let y_t_1_full = y.clone().remove_row(y_len - 1);

    // build Delta[y[t]] = y[t] - y[t-1] by removing the first element of y
    // and subtracting y[t-1].
    // The length of Delta[y[t]] is y_len - 1.
    let delta_y = y.clone().remove_row(0) - &y_t_1_full;

    // we want to return as first element the last y_len - 1 -n elements of Delta[y[t]].
    // we have to remove the first n elements of delta_y.
    let delta_y_output = delta_y.clone().remove_rows(0, n);

    if regression == Regression::ConstantAndTrend {
        //time trend column
        let tt = &y.as_slice()[n..y_len];
    }
    // Now for the second element of the tuple, we want to build a matrix of size (y_len
    // - 1 - n) x (n + 1)

    // Create the empty matrix
    let mut x = DMatrix::zeros(y_len - n - 1, n + 1);

    // - The first column is a column of y[t-1]
    let y_t_1 = y_t_1_full.remove_rows(0, n);
    x.column_mut(0).copy_from(&y_t_1);

    // - The next n columns are shifted elements of Delta[y[t]] (by removing the last element)
    if n > 0 {
        let mut delta_y_shifted = delta_y.clone().remove_row(delta_y.len() - 1);

        for i in 0..n {
            let col = delta_y_shifted.clone().remove_rows(0, n - i - 1);
            x.column_mut(i + 1).copy_from(&col);
            let delta_y_shifted_len = delta_y_shifted.len();
            delta_y_shifted = delta_y_shifted.remove_row(delta_y_shifted_len - 1);
        }
    }

    Ok((delta_y_output.into_owned(), x, y_len - n - 1))
}

#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, Matrix, Vector};

    #[test]
    fn test_prepare() {
        let sz = 10;
        let n = 2;

        let y = vec![1., 3., 6., 10., 15., 21., 28., 36., 45., 55.];
        assert_eq!(y.len(), sz);

        let y = Matrix::from(y);

        let (delta_y, x, sz_) = super::prepare(&y, n).unwrap();

        assert_eq!(sz_, sz - n - 1);

        // Delta[y[t]] = y[t] - y[t-1]
        assert_eq!(delta_y, Vector::from(vec![4., 5., 6., 7., 8., 9., 10.]));

        assert_eq!(x.shape(), (sz - n - 1, n + 1));

        assert_eq!(
            x,
            DMatrix::from_row_slice(
                sz - n - 1,
                n + 1,
                &[
                    // y[t-1], Delta[y[t]].shift(1), Delta[y[t]].shift(2)
                    6., 3., 2., //  row 0
                    10., 4., 3., // row 1
                    15., 5., 4., // row 2
                    21., 6., 5., // row 3
                    28., 7., 6., // row 4
                    36., 8., 7., // row 5
                    45., 9., 8. //  row 6
                ],
            )
        );
    }

    #[test]
    fn test_prepare_minimum_size_00() {
        let n = 0;

        let y: Vec<f32> = vec![];

        let y = Matrix::from(y);

        let res = super::prepare(&y, n);
        assert!(res.is_err());
    }

    #[test]
    fn test_prepare_minimum_size_0() {
        let n = 0;

        let y = vec![1.];

        let y = Matrix::from(y);

        let res = super::prepare(&y, n);
        assert!(res.is_err());
    }

    #[test]
    fn test_prepare_minimum_size_1() {
        let n = 1;

        let y = vec![1.];

        let y = Matrix::from(y);

        let res = super::prepare(&y, n);
        assert!(res.is_err());
    }

    #[test]
    fn test_prepare_minimum_size_2() {
        let n = 1;

        let y = vec![1., 3.];

        let y = Matrix::from(y);

        let res = super::prepare(&y, n);
        assert!(res.is_err());
    }

    #[test]
    fn test_prepare_minimum_size_3() {
        let n = 2;

        let y = vec![1., 3.];

        let y = Matrix::from(y);

        let res = super::prepare(&y, n);
        assert!(res.is_err());
    }
}
