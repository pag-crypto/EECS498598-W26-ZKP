// Comprehensive unit tests for src/poly.rs
// Tests cover Multilinear<F> and Univariate<F> polynomial structures

use p1::moduli::Thirteen;
use p1::poly::{Multilinear, Univariate};
use p1::zq::Zq;
use p1::{One, Zero};
use proptest::prelude::*;
use sfs_bigint::U256;

type F13 = Zq<Thirteen>;

/// Create a Multilinear polynomial from a u64 array for test readability
pub fn ml_from_evals(n_vars: usize, evals: &[u64]) -> Multilinear<F13> {
    Multilinear::new(n_vars, evals.iter().map(|&x| F13::from(x)).collect())
}

/// Create a point from u64 array
pub fn point_from_u64s(coords: &[u64]) -> Vec<F13> {
    coords.iter().map(|&x| F13::from(x)).collect()
}

/// Strategy for generating arbitrary F13 values
pub fn arb_f13() -> BoxedStrategy<F13> {
    any::<U256>().prop_map(F13::new).boxed()
}

/// Strategy for generating arbitrary Multilinear polynomials
pub fn arb_multilinear(max_log_nvars: usize) -> impl Strategy<Value = Multilinear<F13>> {
    (0..=max_log_nvars).prop_flat_map(|n_vars| {
        let size = 1 << n_vars;
        proptest::collection::vec(arb_f13(), size..=size)
            .prop_map(move |evals| Multilinear::new(n_vars, evals))
    })
}

/// Strategy for generating two Multilinear polynomials with the same n_vars
pub fn arb_multilinear_pair(
    max_log_nvars: usize,
) -> impl Strategy<Value = (Multilinear<F13>, Multilinear<F13>)> {
    (0..=max_log_nvars).prop_flat_map(|n_vars| {
        let size = 1 << n_vars;
        (
            proptest::collection::vec(arb_f13(), size..=size),
            proptest::collection::vec(arb_f13(), size..=size),
        )
            .prop_map(move |(evals_a, evals_b)| {
                (
                    Multilinear::new(n_vars, evals_a),
                    Multilinear::new(n_vars, evals_b),
                )
            })
    })
}
/// Strategy for generating arbitrary Univariate polynomials
pub fn arb_univariate(max_degree: usize) -> impl Strategy<Value = Univariate<F13>> {
    (0..=max_degree).prop_flat_map(|degree| {
        proptest::collection::vec(arb_f13(), degree + 1).prop_map(Univariate::new)
    })
}

use proptest::collection::vec as arb_vec;

#[test]
fn test_eq_zero_vars() {
    let g: Vec<F13> = vec![];
    let eq = Multilinear::eq_tilde(&g);

    assert_eq!(eq.evals.len(), 1);
    assert_eq!(eq.evals[0], F13::one(), "eq([]) should be constant 1");
}
#[test]
fn test_eq_one_var_at_zero() {
    let g = vec![F13::zero()];
    let eq = Multilinear::eq_tilde(&g);

    assert_eq!(eq.evals.len(), 2);
    assert_eq!(eq.evals[0], F13::one(), "eq([0]) at x=0 should be 1");
    assert_eq!(eq.evals[1], F13::zero(), "eq([0]) at x=1 should be 0");
}

#[test]
fn test_eq_one_var_at_one() {
    let g = vec![F13::one()];
    let eq = Multilinear::eq_tilde(&g);

    assert_eq!(eq.evals.len(), 2);
    assert_eq!(eq.evals[0], F13::zero(), "eq([1]) at x=0 should be 0");
    assert_eq!(eq.evals[1], F13::one(), "eq([1]) at x=1 should be 1");
}
#[test]
fn test_add_polynomials() {
    let poly1 = ml_from_evals(2, &[1, 2, 3, 4]);
    let poly2 = ml_from_evals(2, &[5, 6, 7, 8]);
    let result = poly1 + poly2;

    assert_eq!(result.evals, [6.into(), 8.into(), 10.into(), 12.into()])
}

#[test]
fn test_evaluate_simple() {
    let poly = ml_from_evals(2, &[5, 5, 5, 5]);
    let point = point_from_u64s(&[7, 11]);
    let result = poly.evaluate(&point);

    assert_eq!(result, F13::from(5),);
}

#[test]
fn test_to_univariate_1var_poly() {
    let poly = ml_from_evals(1, &[3, 7]);
    let uni = poly.to_univariate(0);
    assert_eq!(uni.coeffs(), &[F13::from(3), F13::from(4)]);
}

#[test]
fn test_to_univariate_2vars_first_var() {
    let poly = ml_from_evals(2, &[1, 2, 3, 4]);
    let uni = poly.to_univariate(0);
    let res = poly.partial_eval(&[F13::zero()]) + poly.partial_eval(&[F13::one()]);
    assert_eq!(res.evaluate(&[F13::zero()]), uni.evaluate(F13::zero()));
    assert_eq!(res.evaluate(&[F13::one()]), uni.evaluate(F13::one()));
    assert_eq!(uni.coeffs(), &[F13::from(3), F13::from(4)]);
}

#[test]
fn test_to_univariate_2vars_second_var() {
    let poly = ml_from_evals(2, &[1, 2, 3, 4]);
    let uni = poly.to_univariate(1);
    assert_eq!(uni.coeffs(), &[F13::from(4), F13::from(2)]);
}

#[test]
fn test_to_univariate_3vars_middle_var() {
    let poly = ml_from_evals(3, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let uni = poly.to_univariate(1);
    assert_eq!(uni.coeffs(), &[F13::from(1), F13::from(8)]);
}

#[test]
fn test_to_univariate_zero_poly() {
    let poly = ml_from_evals(2, &[0, 0, 0, 0]);
    let uni = poly.to_univariate(0);
    assert_eq!(uni.coeffs(), &[])
}

#[test]
fn test_partial_eval_empty_point() {
    let poly = ml_from_evals(2, &[1, 2, 3, 4]);
    let result = poly.partial_eval(&[]);

    assert_eq!(
        result, poly,
        "partial_eval with empty point should be identity"
    );
}

#[test]
fn test_partial_eval_single_var_1var_poly() {
    let poly = ml_from_evals(1, &[3, 7]);
    let point = vec![F13::from(5)];
    let result = poly.partial_eval(&point);

    assert_eq!(result.evals.len(), 1, "should reduce to constant");
    // f(5) = 3 + 5*(7-3) = 3 + 5*4 = 3 + 20 = 3 + 7 = 10 (mod 13)
    assert_eq!(result.evals[0], F13::from(10));
}

#[test]
fn test_partial_eval_first_var_2vars() {
    let poly = ml_from_evals(2, &[1, 2, 3, 4]);
    let point = vec![F13::from(0)];
    let result = poly.partial_eval(&point);

    assert_eq!(result.evals.len(), 2, "should reduce from 2 vars to 1 var");
    assert_eq!(result.evals[0], F13::from(1));
    assert_eq!(result.evals[1], F13::from(3));
}

#[test]
fn test_partial_eval_first_var_at_one_2vars() {
    let poly = ml_from_evals(2, &[1, 2, 3, 4]);
    let point = vec![F13::from(1)];
    let result = poly.partial_eval(&point);

    assert_eq!(result.evals.len(), 2);
    assert_eq!(result.evals[0], F13::from(2));
    assert_eq!(result.evals[1], F13::from(4));
}

#[test]
fn test_partial_eval_both_vars_2vars() {
    let poly = ml_from_evals(2, &[1, 2, 3, 4]);
    let point = point_from_u64s(&[0, 0]);
    let result = poly.partial_eval(&point);

    assert_eq!(result.evals.len(), 1, "should reduce to constant");
    assert_eq!(result.evals[0], F13::from(1));
}

#[test]
fn test_partial_eval_all_vars_matches_evaluate() {
    let poly = ml_from_evals(3, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let point = point_from_u64s(&[2, 3, 4]);

    let via_partial = poly.partial_eval(&point).evals[0];
    let via_evaluate = poly.evaluate(&point);

    assert_eq!(
        via_partial, via_evaluate,
        "partial_eval and evaluate should match"
    );
}

#[test]
fn test_partial_eval_sequential_2vars() {
    let poly = ml_from_evals(2, &[1, 2, 3, 4]);

    let point1 = vec![F13::from(2)];
    let point2 = vec![F13::from(3)];

    let seq_result = poly.partial_eval(&point1).partial_eval(&point2).evals[0];
    let direct_result = poly.partial_eval(&point_from_u64s(&[2, 3])).evals[0];

    assert_eq!(
        seq_result, direct_result,
        "sequential partial_eval should match direct"
    );
}

#[test]
fn test_partial_eval_sequential_3vars() {
    let poly = ml_from_evals(3, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let a = F13::from(2);
    let b = F13::from(3);
    let c = F13::from(4);

    let seq_result = poly
        .partial_eval(&[a])
        .partial_eval(&[b])
        .partial_eval(&[c])
        .evals[0];

    let direct_result = poly.partial_eval(&[a, b, c]).evals[0];

    assert_eq!(
        seq_result, direct_result,
        "3-step sequential should match direct"
    );
}

#[test]
fn test_partial_eval_zero_polynomial() {
    let poly = ml_from_evals(2, &[0, 0, 0, 0]);
    let point = vec![F13::from(5)];
    let result = poly.partial_eval(&point);

    for eval in result.evals.iter() {
        assert_eq!(*eval, F13::zero(), "partial_eval of zero should stay zero");
    }
}

#[test]
fn test_partial_eval_constant_polynomial() {
    let poly = ml_from_evals(2, &[7, 7, 7, 7]);
    let point = vec![F13::from(3)];
    let result = poly.partial_eval(&point);

    for eval in result.evals.iter() {
        assert_eq!(
            *eval,
            F13::from(7),
            "partial_eval of constant should stay constant"
        );
    }
}
proptest! {
    #[test]
    fn prop_add_commutative((a, b) in arb_multilinear_pair(5)) {
        let ab = a.clone() + b.clone();
        let ba = b + a;
        prop_assert_eq!(ab, ba);
    }

    #[test]
    fn prop_add_identity(a in arb_multilinear(5)) {
        let n_vars = a.evals.len().ilog2() as usize;
        let zero = Multilinear::new(n_vars, vec![F13::zero(); a.evals.len()]);

        let a_plus_zero = a.clone() + zero.clone();
        let zero_plus_a = zero + a.clone();

        prop_assert_eq!(&a_plus_zero, &a);
        prop_assert_eq!(&zero_plus_a, &a);
    }

    #[test]
    fn prop_sub_inverse(a in arb_multilinear(5)) {
        let n_vars = a.evals.len().ilog2() as usize;
        let zero = Multilinear::new(n_vars, vec![F13::zero(); a.evals.len()]);

        let result = a.clone() - a;
        prop_assert_eq!(result, zero);
    }

    #[test]
    fn prop_neg_involution(a in arb_multilinear(5)) {
        let double_neg = -(-a.clone());
        prop_assert_eq!(double_neg, a);
    }

    #[test]
    fn prop_add_neg_is_sub((a, b) in arb_multilinear_pair(5)) {
        let sub_result = a.clone() - b.clone();
        let add_neg_result = a + (-b);

        prop_assert_eq!(sub_result, add_neg_result);
    }

    #[test]
    fn prop_evaluate_at_boolean_matches_table(poly in arb_multilinear(4)) {
        let n_vars = poly.evals.len().ilog2() as usize;

        // Check all boolean points
        for i in 0..(1 << n_vars) {
            let point: Vec<F13> = (0..n_vars)
                .map(|j| {
                    if (i >> j) & 1 == 1 {
                        F13::one()
                    } else {
                        F13::zero()
                    }
                })
                .collect();

            let result = poly.evaluate(&point);
            prop_assert_eq!(result, poly[i]);
        }
    }

    #[test]
    fn prop_eq_indicator(g_vals in arb_vec(0u8..2, 1..=5)) {
        let g: Vec<F13> = g_vals.iter().map(|&x| F13::from(x as u64)).collect();

        let eq = Multilinear::eq_tilde(&g);

        // eq(g) evaluated at g should be 1 (for boolean g)
        let result = eq.evaluate(&g);
        prop_assert_eq!(result, F13::one());

        // eq(g) evaluated at different boolean point should be 0
        let n_vars = g.len();
        for i in 0..(1 << n_vars) {
            let h: Vec<F13> = (0..n_vars)
                .map(|j| {
                    if (i >> j) & 1 == 1 {
                        F13::one()
                    } else {
                        F13::zero()
                    }
                })
                .collect();

            if h != g {
                let result_h = eq.evaluate(&h);
                prop_assert_eq!(result_h, F13::zero());
            }
        }
    }
    #[test]
    fn prop_partial_eval_dimension_reduction(
        poly in arb_multilinear(5),
        r in arb_f13()
    ) {
        let n_vars = poly.evals.len().ilog2() as usize;
        prop_assume!(n_vars > 0);

        let point = vec![r];
        let result = poly.partial_eval(&point);
        let expected_size = 1 << (n_vars - 1);

        prop_assert_eq!(result.evals.len(), expected_size);
    }

    #[test]
    fn prop_partial_eval_full_equals_evaluate(poly in arb_multilinear(3), point in arb_vec(arb_f13(), 1 << 3)) {
        let n_vars = poly.evals.len().ilog2() as usize;

        let point: Vec<F13> = point.into_iter().take(n_vars).collect();

        let via_partial = poly.partial_eval(&point).evals[0];
        let via_evaluate = poly.evaluate(&point);

        prop_assert_eq!(via_partial, via_evaluate);
    }

    #[test]
    fn prop_univariate_mul_commutative(a in arb_univariate(10), b in arb_univariate(10)) {
        prop_assert_eq!(a.clone() * b.clone(), b * a)
    }

    #[test]
    fn prop_univariate_mul_associative(a in arb_univariate(10), b in arb_univariate(10), c in arb_univariate(10)) {
        prop_assert_eq!(a.clone() * (b.clone() * c.clone()), (a.clone() * b.clone()) * c.clone())
    }

    #[test]
    fn prop_univariate_add_commutative(a in arb_univariate(10), b in arb_univariate(10)) {
        prop_assert_eq!(b.clone() + a.clone(), a + b)
    }

    #[test]
    fn prop_univariate_add_associative(a in arb_univariate(10), b in arb_univariate(10), c in arb_univariate(10)) {
        prop_assert_eq!((a.clone() + b.clone()) + c.clone(), a + (b + c))
    }

    #[test]
    fn prop_univariate_mul_distributes(a in arb_univariate(10), b in arb_univariate(10), c in arb_univariate(10)) {
        prop_assert_eq!(a.clone() * (b.clone() + c.clone()), a.clone() * b + a * c)
    }

    #[test]
    fn prop_add_sub_inverse(a in arb_univariate(10), b in arb_univariate(10)) {
        prop_assert_eq!(a.clone() + b.clone() - a, b)
    }

    #[test]
    fn prop_eval_at_interpolate(evals in arb_vec(arb_f13(), 10)) {
        let points = (0..10).map(F13::from).zip(evals).collect::<Vec<_>>();
        let poly = Univariate::<F13>::interpolate(&points);
        for (x, y) in points {
            prop_assert_eq!(poly.evaluate(x), y)
        }

    }

}
