use std::{
    fmt,
    iter::{self, Product, Sum},
    marker::PhantomData,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    str::FromStr,
};

use num_traits::{Inv, One, Pow, Zero};
use serde::{Deserialize, Serialize};
use sfs_bigint::U256;

use crate::{Field, Random, moduli::PrimeModulus};

/// An integer modulo `Q`, where `Q` is encoded at the type level.
///
/// This struct represents elements of the ring ℤ/Qℤ (integers mod Q). The actual
/// value is stored as a `U256`, but the modulus `Q` is tracked through the type
/// system rather than stored as runtime data. For this course you may assume that moduli are less
/// than 256 bits.
///
/// A note on `PhantomData`:
///
/// We want the type system to distinguish between integers under different moduli.
/// For example, `Zq<Modulus17>` and `Zq<Modulus23>` should be incompatible types.
///
/// However, we don't actually need to *store* the modulus in each element—it's
/// a constant determined by the type.
///
/// It's a zero-sized type that disappears at runtime. It just serves as a marker for the type
/// system.
///
/// Q should implement PrimeModulus which allows its value to be queried via `Q::VALUE`
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Zq<Q> {
    value: U256,

    #[serde(skip)]
    _modulus: PhantomData<Q>,
}

impl<Q> Zq<Q> {
    /// Constructs a Zq<Q> without reducing `value` first. Can be a performance optimization when
    /// you _know_ a value is less than the modulus.
    const fn new_unchecked(value: U256) -> Self {
        Zq {
            value,
            _modulus: PhantomData,
        }
    }

    /// you _must_ ensure when calling this that the integer represented by s is < `Q::VALUE`.
    /// Should only be used when constructing a Zq<Q> in `const` contexts.
    /// (Not relevant for student implementations)
    pub(crate) const fn from_str_unchecked(s: &str, radix: u32) -> Self {
        Self::new_unchecked(U256::from_str_radix_const(s, radix))
    }

    #[must_use]
    pub const fn as_int(&self) -> &U256 {
        &self.value
    }
}

impl<Q: PrimeModulus> Zq<Q> {
    pub fn new(value: U256) -> Self {
        Zq::new_unchecked(value % Q::VALUE)
    }
    // returns *self^2
    pub fn square(&self) -> Self {
        todo!()
    }
    // Returns *self^3
    pub fn cube(&self) -> Self {
        todo!()
    }

    //Note that this will just reduce the bytes mod Q; this does _not_ guarantee a uniform distribution!
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Zq::new(U256::from_le_slice(bytes))
    }

    pub fn legendre_symbol(&self) -> i32 {
        let exponent = (Q::VALUE - 1.into()) / 2.into();
        //This will be 1, -1 (= Q-1), or 0. Figure out which in a dumb way that branches.
        let result = self.pow(exponent);
        if result == Zq::one() {
            1
        } else if result == Zq::new(Q::VALUE - 1.into()) {
            -1
        } else if result == Zq::zero() {
            0
        } else {
            panic!("Result of legendre symbol is not 1, -1, or 0");
        }
    }

    pub fn square_roots(&self) -> Option<(Self, Self)> {
        //First panic if we can't use the easy method. We don't have a tonelli-shanks.
        assert!((Q::VALUE % 4.into()) == 3.into());
        let legendre_symbol = self.legendre_symbol();
        //If we don't have square roots, return None
        if legendre_symbol != 1 {
            return None;
        }
        //Now we know we have square roots and we can use the easy method,
        //so compute them.
        let y0 = self.pow((Q::VALUE + 1.into()) / 4.into());
        let y1 = Zq::new(Q::VALUE - y0.value);
        Some((y0, y1))
    }

    /// Computes the modular inverse of many elements simultaneously.
    ///
    /// Given `[a, b, c, ...]`, returns `[a⁻¹, b⁻¹, c⁻¹, ...]`.
    ///
    /// Montgomery batch inversion allows one to do this with just a single modular inverse
    /// and O(n) multiplications
    ///
    /// # Panics
    ///
    /// Panics if any element is zero.
    pub fn batch_invert(values: &[Zq<Q>]) -> Vec<Zq<Q>> {
        // OPTIONAL: replace me with montgomery batch inversion
        values.iter().map(|val| val.inv()).collect()
    }

    /// Returns the number of bits needed to represent this value.
    pub fn bit_length(&self) -> usize {
        self.value.bit_length()
    }

    /// Returns true if the i-th bit (0-indexed from the LSB) is set.
    pub fn bit(&self, i: usize) -> bool {
        self.value.bit(i)
    }
}

impl<Q: PrimeModulus> FromStr for Zq<Q> {
    type Err = <U256 as FromStr>::Err;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        U256::from_str(s).map(Zq::new)
    }
}
impl<Q: PrimeModulus> Pow<u64> for Zq<Q> {
    type Output = Zq<Q>;
    /// Computes modular exponentiation: `self^exp mod Q`.
    ///
    /// # Hint
    ///
    /// Use the **square-and-multiply** algorithm (analogous to double-and-add
    /// for elliptic curves) to reduce the number of operations from `exp` to
    /// at most `2 · log₂(exp)`.
    fn pow(self, exp: u64) -> Self::Output {
        todo!()
    }
}

impl<Q: PrimeModulus> Pow<U256> for Zq<Q> {
    type Output = Zq<Q>;
    /// Computes modular exponentiation: `self^exp mod Q`.
    ///
    /// This is the same operation as [`Pow<u64>::pow`], but accepts a 256-bit
    /// exponent. Your implementation should look substantially similar.
    fn pow(self, exp: U256) -> Self::Output {
        todo!()
    }
}

impl<Q> From<u64> for Zq<Q>
where
    Q: PrimeModulus,
{
    fn from(value: u64) -> Self {
        Zq::new(From::from(value))
    }
}

impl<Q> From<bool> for Zq<Q>
where
    Q: PrimeModulus,
{
    fn from(value: bool) -> Self {
        Zq::new(U256::from(u64::from(value)))
    }
}
impl<Q: PrimeModulus> Add for Zq<Q> {
    type Output = Zq<Q>;
    /// Computes modular addition: `(self + rhs) mod Q`.
    ///
    /// Returns the unique representative in `[0, Q)` that is congruent to the
    /// sum of `self` and `rhs` modulo `Q`.
    ///
    /// # The Challenge
    ///
    /// Both `self.value` and `rhs.value` are in `[0, Q)`, so their sum lies in
    /// `[0, 2Q - 2]`. The result must be reduced back into `[0, Q)`:
    ///
    /// ```text
    /// result = if sum >= Q::VALUE { sum - Q::VALUE } else { sum }
    /// ```
    ///
    /// However there's a wrinkle!
    ///
    /// `U256` arithmetic can overflow! If `Q` is close to 2²⁵⁶,
    /// then `self + rhs` might exceed 2²⁵⁶ - 1 and wrap around.
    ///
    /// # Useful Primitives
    ///
    /// `U256` provides two methods that help handle overflow and underflow:
    ///
    /// ### `carrying_add`
    /// ```text
    /// fn carrying_add(&self, rhs: &U256) -> (U256, bool)
    /// ```
    /// Computes `self + rhs`, returning:
    /// - The low 256 bits of the sum (wrapping on overflow)
    /// - A boolean indicating whether overflow occurred (i.e., the "carry out")
    ///
    /// Example: if `a + b = 2²⁵⁶ + 42`, then `carrying_add` returns `(42, true)`.
    ///
    /// ### `borrowing_sub`
    /// ```text
    /// fn borrowing_sub(&self, rhs: &U256) -> (U256, bool)
    /// ```
    /// Computes `self - rhs`, returning:
    /// - The result as if computed with unlimited precision, then truncated to 256 bits
    /// - A boolean indicating whether underflow occurred (i.e., a "borrow" was needed)
    ///
    /// Example: if `self = 10` and `rhs = 15`, then `borrowing_sub` returns
    /// `(2²⁵⁶ - 5, true)` since the subtraction underflowed.
    fn add(self, rhs: Self) -> Self::Output {
        todo!()
    }
}
impl<Q: PrimeModulus> Neg for Zq<Q> {
    type Output = Zq<Q>;
    /// Computes the additive inverse: `-self mod Q`.
    fn neg(self) -> Self::Output {
        todo!()
    }
}

impl<Q: PrimeModulus> Neg for &Zq<Q> {
    type Output = Zq<Q>;
    fn neg(self) -> Self::Output {
        -*self
    }
}

impl<Q: PrimeModulus> Zero for Zq<Q> {
    fn zero() -> Self {
        Zq::<Q>::new_unchecked(U256::zero())
    }

    fn is_zero(&self) -> bool {
        self.value.is_zero()
    }
}

impl<Q: PrimeModulus> One for Zq<Q> {
    fn one() -> Self {
        Zq::<Q>::new_unchecked(U256::one())
    }
}
impl<Q: PrimeModulus> Sub for Zq<Q> {
    type Output = Zq<Q>;
    /// Computes modular subtraction: `(self - rhs) mod Q`.
    ///
    /// Returns the unique representative in `[0, Q)` that is congruent to the
    /// difference of `self` and `rhs` modulo `Q`.
    ///
    /// # The Challenge
    ///
    /// Both `self.value` and `rhs.value` are in `[0, Q)`, so their difference lies in
    /// `(-(Q - 1), Q - 1)`, or equivalently `[-(Q - 1), Q - 1]`. When `self < rhs`,
    /// the result is negative and must be corrected:
    ///
    /// ```text
    /// result = if self >= rhs { self - rhs } else { self - rhs + Q::VALUE }
    /// ```
    ///
    /// The tricky part: `U256` is unsigned and cannot represent negative numbers!
    /// Subtracting a larger value from a smaller one causes underflow and wraps around.
    ///
    /// # Useful Primitives
    ///
    /// See the documentation for [`Add::add`](Self::add) for detailed explanations of
    /// `U256::carrying_add` and `U256::borrowing_sub`. The short version:
    ///
    /// - `borrowing_sub` returns `(wrapped_result, underflow_occurred)`
    /// - `carrying_add` returns `(wrapped_result, overflow_occurred)`
    fn sub(self, rhs: Self) -> Self::Output {
        let (tentative_diff, underflow) = self.value.borrowing_sub(&rhs.value);
        Zq::new_unchecked(if underflow {
            tentative_diff.carrying_add(&Q::VALUE).0
        } else {
            tentative_diff
        })
    }
}
impl<Q: PrimeModulus> Mul for Zq<Q> {
    type Output = Zq<Q>;
    /// Computes modular multiplication: `(self * rhs) mod Q`.
    ///
    /// Returns the unique representative in `[0, Q)` that is congruent to the
    /// product of `self` and `rhs` modulo `Q`.
    ///
    /// # The Challenge
    ///
    /// Both `self.value` and `rhs.value` are in `[0, Q)`, so their product lies in
    /// `[0, (Q - 1)²]`. Unlike addition (where the result fits in at most 257 bits),
    /// multiplication of two 256-bit numbers can produce a result up to 512 bits.
    ///
    /// For this, you may use a wider integer type U512 to store the intermediary product.
    ///
    /// # Useful Primitives
    ///
    /// ### `widening_mul`
    /// ```text
    /// fn widening_mul(&self, rhs: &U256) -> U512
    /// ```
    /// Computes the full 512-bit product of two 256-bit integers. No overflow is possible
    /// since the result type is wide enough to hold any product.
    ///
    /// ### `U512::split`
    /// ```text
    /// fn split(&self) -> (U256, U256)
    /// ```
    /// Splits a 512-bit integer into its low and high 256-bit halves: `(low, high)`.
    fn mul(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

impl<Q: PrimeModulus> MulAssign<Zq<Q>> for Zq<Q> {
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl<Q: PrimeModulus> SubAssign<Zq<Q>> for Zq<Q> {
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl<Q: PrimeModulus> AddAssign<Zq<Q>> for Zq<Q> {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl<Q: PrimeModulus> DivAssign<Zq<Q>> for Zq<Q> {
    fn div_assign(&mut self, other: Self) {
        *self = *self / other;
    }
}

impl<Q: PrimeModulus> Div for Zq<Q> {
    type Output = Self;
    #[expect(clippy::suspicious_arithmetic_impl)]
    fn div(self, other: Self) -> Self::Output {
        self * other.inv()
    }
}

impl<Q: PrimeModulus> fmt::Debug for Zq<Q> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} (mod {})", self.value, Q::VALUE)
    }
}

impl<Q: PrimeModulus> fmt::Display for Zq<Q> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}
/// Generates a uniformly random element of ℤ/Qℤ.
///
/// Returns a random value sampled uniformly from `[0, Q)`.
///
/// # Why Not Just Use `random_bytes % Q`?
///
/// A tempting approach is to generate a random 256-bit integer and reduce it mod `Q`:
///
/// ```text
/// let r = U256::random(); // U256 doesn't actually have this method, but suppose it did
/// return r % Q;  // DON'T DO THIS!
/// ```
///
/// This produces a **biased** distribution. To see why, consider a simple example
/// with small numbers: suppose we're sampling mod 3 from a random value in `[0, 7]`:
///
/// ```text
/// value:     0  1  2  3  4  5  6  7
/// value % 3: 0  1  2  0  1  2  0  1
/// ```
///
/// Values 0 and 1 each appear 3 times, but 2 appears only twice. The distribution
/// is not uniform.
///
/// Instead you should utilize `rejection sampling`.
///
/// # Useful Primitives
///
/// ### `rand::Rng::fill_bytes`
/// ```text
/// fn fill_bytes(&mut self, dest: &mut [u8])
/// ```
/// Fills a byte slice with random data. This is the recommended way to generate
/// random bytes from the provided `source`.
///
/// Example usage:
/// ```text
/// let mut bytes = [0u8; 32];
/// source.fill_bytes(&mut bytes);
///
/// ```
/// ### `U256::from_le_slice`
/// ```text
/// fn from_le_slice(bytes: &[u8]) -> U256
/// ```
/// Constructs a `U256` from a byte slice interpreted as a little-endian integer.
/// The slice is zero-padded on the right (high-order bytes) if shorter than 32 bytes.
///
/// ### `U256::bit_length`
/// ```text
/// fn bit_length(&self) -> usize
/// ```
/// Returns the minimum number of bits needed to represent this integer.
/// Equivalently, `⌈log₂(self + 1)⌉` for nonzero values.
impl<Q: PrimeModulus> Random for Zq<Q> {
    fn random(source: &mut impl rand::Rng) -> Self {
        todo!()
    }
}
impl<Q: PrimeModulus> Inv for Zq<Q> {
    type Output = Zq<Q>;

    /// Computes the modular multiplicative inverse: `self⁻¹ mod Q`.
    ///
    /// Returns the unique element `b ∈ [1, Q)` such that `self · b ≡ 1 (mod Q)` via the **extended
    /// Euclidean algorithm**.
    ///
    /// Panics if `self` is zero, or more generally if `gcd(self, Q::VALUE) != 1`
    fn inv(self) -> Self::Output {
        todo!()
    }
}

impl<Q: PrimeModulus> Sum for Zq<Q> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), Add::add)
    }
}

impl<Q: PrimeModulus> Product for Zq<Q> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), Mul::mul)
    }
}

impl<Q: PrimeModulus> Field for Zq<Q> {
    type Order = Q;
}

// STUDENTS NEED NOT PROCEED BELOW THIS LINE
// Note that _in most cases_ you would be able to simply #[derive()]
// all of the below implementations
//
// However, the implementations that #[derive(Trait)] would generate would look like
//
// impl<Q: Trait> Trait for Zq<Q> { .. }
//
// Normally, this is what you want, however because 'Q' is a phantom type (Zq actually contains no
// value of type Q), this implementation is too restrictive. Zq implements e.g. Eq regardless of whether Q does,
// so you want an implementation like
//
// impl<Q> Trait for Zq<Q> { .. }
//
// instead.
//
//
// Derive isn't smart enough to detect this, however, there are crates that automate this kind of boilerplate
// (such as derive_where and derive_more) which provide more powerful versions of Rust's builtin
// `derive`
impl<Q> Copy for Zq<Q> {}
impl<Q> Clone for Zq<Q> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Q> PartialEq for Zq<Q> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<Q> Eq for Zq<Q> {}

impl<Q> PartialOrd for Zq<Q> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<Q> Ord for Zq<Q> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.value.cmp(&other.value)
    }
}

impl<Q> std::hash::Hash for Zq<Q> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}
impl<Q> Default for Zq<Q> {
    fn default() -> Self {
        Zq::new_unchecked(U256::default())
    }
}
