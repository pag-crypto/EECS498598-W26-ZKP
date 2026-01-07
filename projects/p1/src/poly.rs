use std::{
    fmt::{self, Debug},
    iter::{Product, Sum},
    ops::{Add, AddAssign, Index, Mul, MulAssign, Neg, Sub, SubAssign},
    str::FromStr,
};

use itertools::Itertools;

use crate::Field;
use num_traits::Zero;

mod parser;

/// A multilinear polynomial represented by its evaluations over the Boolean hypercube.
///
/// A multilinear polynomial in `n` variables has degree at most 1 in each variable.
/// Such a polynomial is uniquely determined by its 2^n evaluations at all points
/// in {0, 1}^n (the Boolean hypercube).
///
/// # Indexing Convention
///
/// The evaluation at a Boolean point (b_0, b_1, ..., b_{n-1}) is stored at index
/// `i = b_0 · 2^0 + b_1 · 2^1 + ... + b_{n-1} · 2^{n-1}`. In other words, the j-th
/// bit of the index `i` corresponds to the value of variable x_j.
///
/// # Example
///
/// For a polynomial `p(x_0, x_1)` with two variables:
///
/// | Index | Binary | Evaluation  |
/// |-------|--------|-------------|
/// |   0   |   00   | p(0, 0)     |
/// |   1   |   01   | p(1, 0)     |
/// |   2   |   10   | p(0, 1)     |
/// |   3   |   11   | p(1, 1)     |
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Multilinear<F> {
    /// The number of variables in the polynomial.
    n_vars: usize,

    /// Evaluation table of length 2^n_vars.
    ///
    /// `evals[i]` holds the polynomial's value at the Boolean point whose
    /// coordinate vector corresponds to the binary representation of `i`
    /// (least significant bit = variable x_0).
    pub evals: Vec<F>,
}

impl<F: Field> Multilinear<F> {
    pub fn new(n_vars: usize, evals: Vec<F>) -> Self {
        assert!(n_vars < 64);
        assert_eq!(evals.len(), 1 << n_vars);

        Self { n_vars, evals }
    }

    /// Constructs the multilinear extension of the equality polynomial, ẽq(x, g).
    ///
    /// The equality polynomial `eq(x, y)` evaluates to 1 when `x = y` (for binary vectors)
    /// and 0 otherwise. Its multilinear extension over a field is defined as:
    ///
    /// ```text
    /// ẽq(x, g) = ∏_{i=0}^{n-1} (x_i · g_i + (1 - x_i) · (1 - g_i))
    /// ```
    ///
    /// This function returns the multilinear polynomial whose evaluation table contains
    /// ẽq(b, g) for all binary vectors b ∈ {0, 1}^n, where n = g.len().
    ///
    /// At each Boolean evaluation point b:
    /// - If bit i of b is 1: contributes factor `g[i]`
    /// - If bit i of b is 0: contributes factor `1 - g[i]`
    ///
    pub fn eq_tilde(g: &[F]) -> Self {
        todo!()
    }
    /// Evaluates the multilinear polynomial at an arbitrary point in F^n.
    ///
    /// Given a point `r = (r_0, r_1, ..., r_{n-1})` with coordinates in the field F,
    /// computes the unique value of the multilinear polynomial at that point.
    ///
    /// # Panics
    /// This function panics if `self.n_vars != point.len()`
    pub fn evaluate(&self, point: &[F]) -> F {
        assert!(self.n_vars == point.len());
        todo!()
    }

    /// Partially evaluates the polynomial by fixing the first `k` variables.
    ///
    /// Given a multilinear polynomial `p(x_0, x_1, ..., x_{n-1})` and a partial point
    /// `r = (r_0, r_1, ..., r_{k-1})`, computes the restricted polynomial:
    ///
    /// ```text
    /// q(x_k, ..., x_{n-1}) = p(r_0, r_1, ..., r_{k-1}, x_k, ..., x_{n-1})
    /// ```
    ///
    /// The result is a multilinear polynomial in `n - k` variables.
    ///
    /// # Panics
    /// This function panics if `partial_point.len() > self.n_vars`
    pub fn partial_eval(&self, partial_point: &[F]) -> Self {
        assert!(partial_point.len() <= self.n_vars);
        todo!()
    }

    /// Computes the univariate polynomial obtained by summing over all variables except one.
    ///
    /// Given a multilinear polynomial `p(x_0, x_1, ..., x_{n-1})`, this function treats
    /// the variable at `variable_index` as a free variable `X` and sums over all Boolean
    /// assignments to the remaining variables:
    ///
    /// ```text
    /// g(X) = Σ_{b ∈ {0,1}^{n-1}} p(b_0, ..., b_{j-1}, X, b_{j+1}, ..., b_{n-1})
    /// ```
    ///
    /// where `j = variable_index`.
    pub fn to_univariate(&self, variable_index: usize) -> Univariate<F> {
        todo!()
    }
}

impl<F: Field> Add for Multilinear<F> {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self::Output {
        self += &rhs;
        self
    }
}

impl<F: Field> AddAssign<&Multilinear<F>> for Multilinear<F> {
    /// Adds another multilinear polynomial to `self` in place.
    ///
    /// Computes `self := self + rhs`, where addition is pointwise over the
    /// evaluation tables.
    ///
    /// # Panics
    ///
    /// Panics if `self.n_vars != rhs.n_vars`.
    fn add_assign(&mut self, rhs: &Self) {
        assert_eq!(self.n_vars, rhs.n_vars);
        todo!()
    }
}
impl<F: Field> SubAssign<&Multilinear<F>> for Multilinear<F> {
    /// Subtracts another multilinear polynomial from `self` in place.
    ///
    /// Computes `self := self - rhs`, where subtraction is pointwise over the
    /// evaluation tables.
    ///
    /// # Panics
    ///
    /// Panics if `self.n_vars != rhs.n_vars`.
    fn sub_assign(&mut self, rhs: &Self) {
        assert_eq!(self.n_vars, rhs.n_vars);
        todo!()
    }
}

impl<F: Field> Sub for Multilinear<F> {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= &rhs;
        self
    }
}

impl<F: Field> Index<usize> for Multilinear<F> {
    type Output = F;
    fn index(&self, index: usize) -> &Self::Output {
        &self.evals[index]
    }
}

impl<F: Field> Neg for Multilinear<F> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self::new(self.n_vars, self.evals.into_iter().map(Neg::neg).collect())
    }
}

/// A univariate polynomial over a field F, represented by its coefficients.
///
/// The polynomial is stored as a vector of coefficients in ascending order of degree:
/// `coeffs[i]` is the coefficient of `x^i`.
///
/// # Example
///
/// The polynomial `p(x) = 2x² + 3x + 1` is represented as:
///
/// | Index | Coefficient | Term  |
/// |-------|-------------|-------|
/// |   0   |      1      |  1    |
/// |   1   |      3      |  3x   |
/// |   2   |      2      |  2x²  |
///
/// giving `coeffs = [1, 3, 2]`.
///
/// # Representation Invariant
///
/// **The coefficient vector must not contain trailing zeros.**
///
/// This ensures a canonical representation: each polynomial has exactly one valid
/// coefficient vector. For example, `2x² + 3x + 1` must be stored as `[1, 3, 2]`,
/// never as `[1, 3, 2, 0]` or `[1, 3, 2, 0, 0]`, etc.
///
/// Consequences of this invariant:
/// - The zero polynomial is represented as `coeffs = []` (empty vector), not `[0]`
/// - `coeffs.len()` equals `degree + 1` for non-zero polynomials
/// - Two polynomials are equal if and only if their coefficient vectors are equal
///
/// # Maintaining the Invariant
///
/// Use the provided [`trim`](Self::trim) method to remove trailing zeros
/// after any operation that might introduce them. It is your responsibility to call
/// `normalize` where appropriate in your implementations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Univariate<F> {
    // coeffs[i] corresponds to the coefficient in front of x^i
    // trailing zeros should always be trimmed
    coeffs: Vec<F>,
}

impl<F: Field> Univariate<F> {
    pub fn new(coeffs: Vec<F>) -> Self {
        let mut ret = Self { coeffs };
        ret.trim();
        ret
    }

    pub fn degree(&self) -> usize {
        self.coeffs.len().saturating_sub(1)
    }

    pub fn coeffs(&self) -> &[F] {
        &self.coeffs
    }

    // remove trailing zeros
    fn trim(&mut self) {
        let new_len = self
            .coeffs
            .iter()
            .rposition(|p| !p.is_zero())
            .map(|pos| pos + 1)
            .unwrap_or(0);
        self.coeffs.truncate(new_len);
    }

    fn constant(c: F) -> Self {
        if c.is_zero() {
            Self::zero()
        } else {
            Self { coeffs: vec![c] }
        }
    }

    /// Constructs the unique polynomial that passes through a given set of points.
    ///
    /// Given `n` points `{(x_0, y_0), (x_1, y_1), ..., (x_{n-1}, y_{n-1})}` with distinct
    /// x-coordinates, there exists a unique polynomial `p(x)` of degree at most `n - 1`
    /// such that `p(x_i) = y_i` for all `i`.
    pub fn interpolate(points: &[(F, F)]) -> Self {
        assert!(!points.is_empty(), "need at least one point");
        todo!()
    }

    /// Computes `p(x) = c_0 + c_1·x + c_2·x² + ... + c_{n-1}·x^{n-1}` for the
    /// given value of `x`.
    ///
    /// # Hints
    ///
    /// Horner's method rewrites the polynomial to minimize multiplications:
    ///
    /// ```text
    /// p(x) = c_0 + x·(c_1 + x·(c_2 + ... + x·(c_{n-2} + x·c_{n-1})...))
    /// ```
    pub fn evaluate(&self, x: F) -> F {
        todo!()
    }
}

impl<F: Field> Zero for Univariate<F> {
    fn zero() -> Self {
        Self { coeffs: vec![] }
    }
    fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }
}

impl<F: Field> Sum for Univariate<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(Add::add).unwrap_or_else(Self::zero)
    }
}

impl<F: Field> Product for Univariate<F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(Mul::mul)
            .unwrap_or_else(|| Self::constant(F::one()))
    }
}

impl<F: Field> fmt::Display for Univariate<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Univariate(")?;

        let mut terms = Vec::new();

        for (i, coeff) in self.coeffs.iter().enumerate().filter(|(_, x)| !x.is_zero()) {
            let term = match i {
                0 => format!("{coeff}"),
                1 => format!("{coeff}*x"),
                _ => format!("{coeff}*x^{i}"),
            };
            terms.push(term);
        }

        if terms.is_empty() {
            write!(f, "0")?;
        } else {
            write!(f, "{}", terms.join(" + "))?;
        }

        write!(f, ")")
    }
}

impl<F: Field> AddAssign<&Univariate<F>> for Univariate<F> {
    fn add_assign(&mut self, other: &Self) {
        // Extend self.coeffs if needed
        if self.coeffs.len() < other.coeffs.len() {
            self.coeffs.resize(other.coeffs.len(), F::zero());
        }

        // Add coefficients
        for (i, &other_coeff) in other.coeffs.iter().enumerate() {
            self.coeffs[i] += other_coeff;
        }
        self.trim();
    }
}

impl<F: Field> Add for Univariate<F> {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self::Output {
        self += &rhs;
        self
    }
}

impl<F: Field> SubAssign<&Univariate<F>> for Univariate<F> {
    fn sub_assign(&mut self, other: &Self) {
        // Extend self.coeffs if needed
        if self.coeffs.len() < other.coeffs.len() {
            self.coeffs.resize(other.coeffs.len(), F::zero());
        }

        // Add coefficients
        for (i, &other_coeff) in other.coeffs.iter().enumerate() {
            self.coeffs[i] -= other_coeff;
        }
        self.trim();
    }
}

impl<F: Field> Sub for Univariate<F> {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= &rhs;
        self
    }
}

impl<F: Field> MulAssign<&Univariate<F>> for Univariate<F> {
    /// Multiplies `self` by another polynomial in place.
    ///
    /// Computes `self := self * other` using polynomial multiplication. If `self`
    /// has degree `n` and `other` has degree `m`, the result has degree `n + m`.
    ///
    /// # Hint
    ///
    /// The coefficient of `x^k` in the product is `Σ_{i+j=k} self[i] * other[j]`.
    fn mul_assign(&mut self, other: &Self) {
        todo!()
    }
}

impl<F: Field> Mul for Univariate<F> {
    type Output = Self;
    fn mul(mut self, rhs: Self) -> Self::Output {
        self *= &rhs;
        self
    }
}

impl<F: Field> MulAssign<F> for Univariate<F> {
    /// Multiplies `self` by a scalar in place.
    ///
    /// Computes `self := c * self`, scaling every coefficient by `c`.
    fn mul_assign(&mut self, rhs: F) {
        todo!()
    }
}

impl<F: Field> Mul<F> for Univariate<F> {
    type Output = Self;
    fn mul(mut self, rhs: F) -> Self::Output {
        self *= rhs;
        self
    }
}

impl<F: Field> Neg for Univariate<F> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            coeffs: self.coeffs.into_iter().map(Neg::neg).collect(),
        }
    }
}

impl<F: Field> Neg for &Univariate<F> {
    type Output = Univariate<F>;
    fn neg(self) -> Self::Output {
        -self.clone()
    }
}

impl<F: Field> FromStr for Univariate<F> {
    type Err = String;

    /// Parses a univariate polynomial from a string.
    ///
    /// Format: `"constant + coeff*x + coeff*x^k + ..."`
    ///
    /// The variable can be written as `x` or `x_0`.
    /// Powers are written as `^k`. If omitted, power is 1.
    ///
    /// # Examples
    /// - `"3 + 2*x + 5*x^2"` → 3 + 2x + 5x²
    /// - `"x^3 - 2*x + 1"` → 1 - 2x + x³
    /// - `"5"` → constant polynomial 5
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let terms = parser::parse_terms(s)?;

        if terms.is_empty() {
            return Ok(Self::constant(F::zero()));
        }

        // Find the maximum degree
        let max_degree = terms
            .iter()
            .map(|term| term.vars.iter().map(|(_, p)| *p).sum::<u64>())
            .max()
            .unwrap_or(0) as usize;

        let mut coeffs = vec![F::zero(); max_degree + 1];

        for term in terms {
            // For univariate, all variables should be x_0 (or just x)
            // The degree is the sum of all powers (which should all be for the same variable)
            if let Some((var_idx, _)) = term.vars.iter().find(|&&(var_idx, _)| var_idx != 0) {
                return Err(format!(
                    "Univariate polynomial cannot have variable x_{var_idx}, only x or x_0",
                ));
            }

            let degree: usize = term.vars.iter().map(|(_, p)| *p as usize).sum();
            let mut val = F::from(term.coeff);
            if !term.sign {
                val = -val;
            }
            coeffs[degree] += val;
        }
        Ok(Self::new(coeffs))
    }
}

impl<F: Field> FromStr for Multilinear<F> {
    type Err = String;

    /// Parses a multilinear polynomial from a string.
    ///
    /// Format: `"constant + coeff*x_i + coeff*x_i*x_j + ..."`
    ///
    /// Each variable can appear at most once per term (degree at most 1).
    /// Variables are written as `x_i` where `i` is the variable index.
    ///
    /// # Examples
    /// - `"2 + 3*x_0 + 4*x_1 + 5*x_0*x_1"` → 2 + 3x₀ + 4x₁ + 5x₀x₁
    /// - `"x_0 + x_1"` → x₀ + x₁
    /// - `"1 - x_0*x_1*x_2"` → 1 - x₀x₁x₂
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let terms = parser::parse_terms(s)?;

        if terms.is_empty() {
            return Ok(Self::new(0, vec![F::zero()]));
        }

        // Determine the number of variables (max index + 1)
        let n_vars = terms
            .iter()
            .flat_map(|term| term.vars.iter().map(|(idx, _)| *idx))
            .max()
            .map_or(0, |m| m + 1);

        // Validate multilinear requirement and no duplicate variables
        for term in &terms {
            for &(var_idx, power) in &term.vars {
                if power > 1 {
                    return Err(format!(
                        "Multilinear polynomial cannot have x_{var_idx}^{power}, maximum power is 1",
                    ));
                }
            }

            if term.vars.iter().map(|(v, _)| *v).unique().count() != term.vars.len() {
                return Err(
                    "Multilinear polynomial cannot have repeated variables in a term".into(),
                );
            }
        }

        // Build the evaluation table
        let num_evals = 1 << n_vars;
        let mut evals = vec![F::zero(); num_evals];

        for term in terms {
            let val = if term.sign {
                F::from(term.coeff)
            } else {
                -F::from(term.coeff)
            };

            let required_bits: usize = term.vars.iter().map(|(v, _)| 1 << v).sum();
            for (b, eval) in evals.iter_mut().enumerate() {
                if (b & required_bits) == required_bits {
                    *eval += val;
                }
            }
        }

        Ok(Self::new(n_vars, evals))
    }
}
