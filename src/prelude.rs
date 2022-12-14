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

//! unit_root is a library for testing for unit roots in time series.
//! This is the public API. Enjoy!

/// Re-export what we need from nalgebra
pub mod nalgebra {
    pub use nalgebra::DVector;
}

/// Errors
pub use crate::Error;

/// Tools
pub mod tools {
    pub use crate::tools::Report;

    /// Dickey-Fuller test
    pub mod dickeyfuller {
        pub use crate::tools::dickeyfuller::constant_no_trend_test;
    }

    /// Augmented Dickey-Fuller test
    pub mod adf {
        pub use crate::tools::adf::constant_no_trend_test;
    }
}

/// Distributions
pub mod distrib {
    /// Dickey-Fuller distribution
    pub mod dickeyfuller {
        pub use crate::distrib::dickeyfuller::constant_no_trend_critical_value;
    }
    pub use crate::distrib::AlphaLevel;
}
