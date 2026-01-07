use std::{
    iter::Sum,
    ops::{Add, Mul, Neg, Sub},
    str::FromStr,
};

use crate::{
    moduli::{P256, P256CurveOrder},
    zq::Zq,
};

use num_traits::Zero;

///The prime defining the underlying field of the curve.
/// Note that this prime is equal to three mod four,
/// so square roots are easy in the field.
pub const SECP256R1_P256: &str =
    "0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF";

/// Curve parameters for the curve equation: y^2 = x^3 + a256*x +b256
pub const SECP256R1_A256: Zq<P256> = Zq::from_str_unchecked(
    "ffffffff00000001000000000000000000000000fffffffffffffffffffffffc",
    16,
);

pub const SECP256R1_B256: Zq<P256> = Zq::from_str_unchecked(
    "5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B",
    16,
);

pub const SECP256R1_CURVE_ORDER: Zq<P256> = Zq::from_str_unchecked(
    "FFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551",
    16,
);

pub const SECP256R1_G_X: Zq<P256> = Zq::from_str_unchecked(
    "6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296",
    16,
);

pub const SECP256R1_G_Y: Zq<P256> = Zq::from_str_unchecked(
    "4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5",
    16,
);

pub const SECP256R1_G: P256Point = P256Point::point_unchecked(SECP256R1_G_X, SECP256R1_G_Y);

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct PrivateZST;

/// A point on the NIST P-256 elliptic curve.
///
/// The P-256 curve is defined by the equation:
///
/// ```text
/// y² = x³ + ax + b  (mod p)
/// ```
///
/// where `a`, `b`, and `p` are specific constants defined by NIST. Points on the
/// curve, together with a special "point at infinity," form a **group** under
/// elliptic curve addition.
///
/// # The Point at Infinity
///
/// Every elliptic curve group includes a point at infinity, denoted `∞` or `𝒪`,
/// which serves as the **identity element**:
///
/// # Preventing Invalid Points (a note on PrivateZST)
///
/// Not every `(x, y)` pair lies on the curve!
///
/// We want to:
/// 1. **Prevent** external code from constructing `Point` with arbitrary coordinates
/// 2. **Allow** external code to pattern match on `Point` to extract `x` and `y` in an ergonomic
///    way.
///
///
/// One solution to this is to use `PrivateZST`: a **zero-sized private type** as a field:
///
/// ```text
/// Point {
///     x: Zq<P256>,    // Readable via pattern matching
///     y: Zq<P256>,    // Readable via pattern matching
///     _priv: PrivateZST,  // Can't be constructed externally
/// }
/// ```
///
/// External code *can* pattern match (ignoring `_priv` with `..`):
///
/// ```text
/// match point {
///     P256Point::Inf => { /* handle infinity */ }
///     P256Point::Point { x, y, .. } => { /* use x and y */ }
/// }
/// ```
///
/// But external code *cannot* construct the variant:
///
/// ```text
/// // Won't compile outside this module, can't name or create PrivateZST
/// let p = P256Point::Point { x, y, _priv: ??? };
/// ```
///
/// Since `PrivateZST` is zero-sized, it adds no runtime overhead. `P256Point::Point`
/// is exactly the size of two field elements.
///
/// An alternative representation would have been to represent `P256Point` as an enum wrapped in a
/// struct with an appropriate set of getter/setter-type methods.
///
/// Use the [`P256Point::point`] constructor to create valid points; it checks that
/// the coordinates satisfy the curve equation.
#[derive(Copy, Clone, Eq, PartialEq, Default)]
pub enum P256Point {
    ///The point at infinity
    #[default]
    Inf,
    ///The coordinates of a non-infinite curve point.
    /// Outside this module, this variant can be constructed via
    /// [`P256Point::point`]
    Point {
        x: Zq<P256>,
        y: Zq<P256>,
        // See above!
        #[allow(private_interfaces)]
        _priv: PrivateZST,
    },
}

impl P256Point {
    pub const GENERATOR: P256Point = SECP256R1_G;

    /// Attempts to construct a `P256Point::Point` from `x` and `y`
    ///
    /// # Errors
    /// Returns `Err(InvalidPointError::NotOnCurve(x, y))` if `(x,y)` is not a curve point.
    #[inline]
    pub fn point(x: Zq<P256>, y: Zq<P256>) -> Result<Self, InvalidPointError> {
        if Self::is_on_curve(&x, &y) {
            Ok(Self::point_unchecked(x, y))
        } else {
            Err(InvalidPointError::NotOnCurve(x, y))
        }
    }

    // convenience shorthand constructor
    // should only be called when x and y are known to be on the curve
    const fn point_unchecked(x: Zq<P256>, y: Zq<P256>) -> Self {
        P256Point::Point {
            x,
            y,
            _priv: PrivateZST,
        }
    }

    /// Returns a generator for the curve from a seed.
    /// The seed is given to an RNG that is used with P256Point::random()
    /// to construct the generator. Note the small size of the
    /// seed---this function should only be used to generate public parameters.
    /// If a protocol needs to generate an unpredictable group element, a
    /// much larger and truly random seed (and a different method) should be used.
    pub fn get_generator_from_seed(seed: usize) -> Self {
        use sha3::{
            Shake128,
            digest::{ExtendableOutput, Update, XofReader},
        };

        const MAX_ATTEMPTS: usize = 50;
        let mut hasher = Shake128::default();
        hasher.update(&seed.to_le_bytes());
        let mut reader = hasher.finalize_xof();
        let mut bytes = [0u8; 33];

        for _ in 0..MAX_ATTEMPTS {
            reader.read(&mut bytes);
            //This function's return value is not exactly uniform over
            //P256, but the statistical distance from uniform is quite small.
            let x = Zq::<P256>::from_bytes(&bytes[..32]);

            let rhs = x.cube() + SECP256R1_A256 * x + SECP256R1_B256;
            //Check if there's ys so that this x is on the curev.
            //We do this in the "slow" way for pedagogical purposes: first, compute the RHS of the curve equation,
            //and check its Legendre symbol to see whether there are square roots.
            //If there are, compute the square roots and choose one of them at random.
            //If there aren't, we go back above and get more bytes for another try at x and do it again.
            //This loop will terminate in only a couple attempts w.h.p.
            //https://papers.mathyvanhoef.com/dragonblood.pdf
            if let Some((y0, y1)) = rhs.square_roots() {
                let bit_choice = (bytes[32] & 1) == 1;
                let y = if bit_choice { y0 } else { y1 };
                return P256Point::point_unchecked(x, y);
            }
        }
        panic!("Could not find a point on the curve after {MAX_ATTEMPTS} attempts");
    }

    /// Returns true iff (x,y) is on the curve. Which is to say, y^2 = x^3 + a256 * x + b256
    #[inline]
    pub fn is_on_curve(x: &Zq<P256>, y: &Zq<P256>) -> bool {
        todo!()
    }

    /// Computes a multi-scalar multiplication (MSM), also known as a linear combination of points.
    ///
    /// Given scalars `(s_0, s_1, ..., s_{n-1})` and elliptic curve points `(P_0, P_1, ..., P_{n-1})`,
    /// computes:
    ///
    /// ```text
    /// s_0·P_0 + s_1·P_1 + ... + s_{n-1}·P_{n-1}
    /// ```
    ///
    /// where `·` denotes scalar multiplication (repeated point addition) and `+` denotes
    /// elliptic curve point addition.
    ///
    /// # Efficiency Considerations
    ///
    /// A naive implementation computes each `s_i·P_i` independently and sums the results.
    /// If each scalar is `k` bits, this requires roughly `n·k` point additions.
    ///
    /// More sophisticated algorithms (such as Pippenger's algorithm or signed-digit
    /// methods) can significantly reduce the number of group operations by sharing
    /// work across the scalar multiplications. For this assignment, a correct naive
    /// implementation is acceptable.
    ///
    /// # Arguments
    ///
    /// * `scalars` - A slice of scalar field elements
    /// * `bases` - A slice of elliptic curve points (the "bases" being scaled)
    ///
    /// # Returns
    ///
    /// The elliptic curve point equal to the linear combination `Σᵢ scalars[i] · bases[i]`.
    ///
    /// # Panics
    ///
    /// Panics if `scalars.len() != bases.len()`.
    #[inline]
    pub fn msm(scalars: &[Zq<P256CurveOrder>], bases: &[P256Point]) -> P256Point {
        assert_eq!(scalars.len(), bases.len());
        todo!()
    }
}

impl Add for P256Point {
    type Output = P256Point;

    /// Computes elliptic curve point addition: `P + Q`.
    ///
    /// This is the group operation on the curve. The geometric interpretation:
    /// draw a line through `P` and `Q`, find the third intersection with the
    /// curve, then reflect across the x-axis.
    ///
    /// # Cases to Handle
    ///
    /// - `P + ∞ = ∞ + P = P` (identity)
    /// - `P + (-P) = ∞` (inverse)
    /// - `P + P` requires the **point doubling** formula (tangent line)
    /// - `P + Q` where `P ≠ ±Q` uses the **chord** formula
    ///
    /// The formulas for the chord and tangent cases differ—make sure to
    /// distinguish when `P = Q` vs `P ≠ Q`.
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

impl Sub for P256Point {
    type Output = P256Point;

    /// Computes elliptic curve point subtraction: `P - Q`.
    ///
    /// # Hint
    ///
    /// This can be implemented trivially using `Add` and `Neg`.
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        todo!()
    }
}
impl Neg for P256Point {
    type Output = P256Point;

    /// Computes the additive inverse: `-P`.
    #[inline]
    fn neg(self) -> Self::Output {
        todo!()
    }
}

impl Neg for &P256Point {
    type Output = P256Point;

    #[inline]
    fn neg(self) -> Self::Output {
        -*self
    }
}

impl Mul<Zq<P256CurveOrder>> for P256Point {
    type Output = P256Point;
    /// Computes scalar multiplication: `s · P`.
    ///
    /// Returns the point `P` added to itself `s` times:
    ///
    /// ```text
    /// s · P = P + P + ... + P   (s times)
    /// ```
    ///
    /// # Hints
    ///
    /// A naive loop is far too slow—scalars can be up to ~2²⁵⁶. Use the
    /// **double-and-add** algorithm, which is analogous to square-and-multiply
    /// for exponentiation:
    ///
    /// - "squaring" → point doubling (`P + P`)
    /// - "multiplying" → point addition (`acc + P`)
    ///
    /// This reduces the number of operations to at most `2k` where `k` is the
    /// bit-length of `s`.
    #[inline]
    fn mul(self, rhs: Zq<P256CurveOrder>) -> Self::Output {
        todo!()
    }
}

impl Sum for P256Point {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(P256Point::zero(), Add::add)
    }
}

impl Zero for P256Point {
    #[inline]
    fn zero() -> Self {
        P256Point::Inf
    }
    #[inline]
    fn is_zero(&self) -> bool {
        matches!(self, P256Point::Inf)
    }
}

// we write our own Debug impl so PrivateZST doesn't show up
impl std::fmt::Debug for P256Point {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Inf => write!(f, "P256Point::Inf"),
            Self::Point { x, y, .. } => f
                .debug_struct("P256Point::Point")
                .field("x", x)
                .field("y", y)
                .finish(),
        }
    }
}

impl std::fmt::Display for P256Point {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Inf => write!(f, "(∞, ∞)"),
            Self::Point { x, y, .. } => write!(f, "({x}, {y})"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum InvalidPointError {
    NotOnCurve(Zq<P256>, Zq<P256>),
    Other(String),
}

impl From<String> for InvalidPointError {
    fn from(value: String) -> Self {
        InvalidPointError::Other(value)
    }
}
impl From<&str> for InvalidPointError {
    fn from(value: &str) -> Self {
        InvalidPointError::Other(value.to_owned())
    }
}

impl std::fmt::Display for InvalidPointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotOnCurve(x, y) => write!(f, "Invalid point: ({x}, {y}) is not on-curve"),
            Self::Other(msg) => write!(f, "Invalid point: {msg}"),
        }
    }
}

impl std::error::Error for InvalidPointError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

impl FromStr for P256Point {
    type Err = InvalidPointError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.is_empty() {
            return Err("Empty string".into());
        }

        //Parse the string as a tuple of two strings
        let (x, y) = s.split_once(',').ok_or("Invalid format")?;

        if x.trim() == "inf" && y.trim() == "inf" {
            return Ok(P256Point::zero());
        }

        //Parse the x and y coordinates
        let x = Zq::<P256>::from_str(x).map_err(|e| format!("{e}"))?;
        let y = Zq::<P256>::from_str(y).map_err(|e| format!("{e}"))?;
        //Check if point is on curve
        Self::point(x, y)
    }
}
