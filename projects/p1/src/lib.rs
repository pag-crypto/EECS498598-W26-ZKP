#![warn(clippy::all)]

pub mod curve;
pub mod moduli;
pub mod poly;
pub mod zq;
pub use num_traits::{One, Pow, Zero};

use serde::{Serialize, de::DeserializeOwned};
use std::{
    fmt::{Debug, Display},
    hash::Hash,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// Interface for generating a (uniformly random) element
/// given a source of randomness
pub trait Random {
    fn random(rng: &mut impl rand::Rng) -> Self;
}

/// Catch-all trait for 'Field' elements in this course
/// Obviously more expansive than the mathematical definition of a field
/// Not all these traits are relevant for this project, you do not have to understand
/// the ones you are not asked to implement!
pub trait Field:
    Copy
    + Zero
    + One
    + Eq
    + Ord
    + Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + AddAssign<Self>
    + SubAssign<Self>
    + MulAssign<Self>
    + DivAssign<Self>
    + From<u64>
    + From<bool>
    + Sum
    + Product
    + Default
    + Pow<u64, Output = Self>
    + Display
    + Debug
    + Send
    + Sync
    + Hash
    + Serialize
    + DeserializeOwned
    + 'static
{
    type Order: moduli::PrimeModulus;
}
