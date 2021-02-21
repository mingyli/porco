use std::{convert::TryFrom, ops};

/// [`Probability`] is a light container for probabilities.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Probability(pub f64);

impl Probability {
    pub const ZERO: Probability = Probability(0.0);
    pub const ONE: Probability = Probability(1.0);
}

impl Eq for Probability {}

impl From<Probability> for f64 {
    fn from(probability: Probability) -> Self {
        probability.0
    }
}

impl From<f64> for Probability {
    fn from(p: f64) -> Self {
        Probability::try_from(p).expect("The probability is between 0.0 and 1.0")
    }
}

// TODO: Consider using TryFrom instead of From.
// impl TryFrom<f64> for Probability {
//     type Error = &'static str;
//
//     fn try_from(p: f64) -> Result<Self, Self::Error> {
//         if (0.0..=1.0).contains(&p) {
//             Ok(Probability(p))
//         } else {
//             Err("TODO: Use error type.")
//         }
//     }
// }

impl ops::Add for Probability {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl ops::Sub for Probability {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

impl ops::Mul for Probability {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self(self.0 * other.0)
    }
}

impl ops::Div for Probability {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Self(self.0 / other.0)
    }
}

impl ops::Div<f64> for Probability {
    type Output = Self;

    fn div(self, other: f64) -> Self {
        Self(self.0 / other)
    }
}
