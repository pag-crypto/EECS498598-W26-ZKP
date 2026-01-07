use num_traits::{Inv, One, Pow, Zero};
use p1::moduli::{P256, Thirteen};
use p1::zq::Zq;
use proptest::prelude::*;
use sfs_bigint::U256;
use std::str::FromStr;

// Strategy functions for generating arbitrary Zq values
// Note: this is a potentially biased distribution, but it's fine for testing purposes

fn arb_zq<Q: p1::moduli::PrimeModulus>() -> BoxedStrategy<Zq<Q>> {
    any::<U256>().prop_map(|value| Zq::new(value)).boxed()
}

fn arb_zq_p256() -> BoxedStrategy<Zq<P256>> {
    arb_zq::<P256>()
}

fn arb_zq_thirteen() -> BoxedStrategy<Zq<Thirteen>> {
    arb_zq::<Thirteen>()
}

#[test]
fn test_square() {
    let z = Zq::<Thirteen>::from(4);
    assert_eq!(z.square(), Zq::<Thirteen>::from(3));
}

#[test]
fn test_cube() {
    let z = Zq::<Thirteen>::from(4);
    assert_eq!(z.cube(), Zq::<Thirteen>::from(12));
}

#[test]
fn test_from_str() {
    let z = Zq::<Thirteen>::from_str("10").unwrap();
    assert_eq!(z, Zq::<Thirteen>::from(10));
}

#[test]
fn test_batch_inversion() {
    let values = vec![
        Zq::<Thirteen>::from(1),
        Zq::<Thirteen>::from(2),
        Zq::<Thirteen>::from(3),
        Zq::<Thirteen>::from(4),
    ];
    let inverted = Zq::<Thirteen>::batch_invert(&values);
    assert_eq!(
        inverted,
        vec![
            Zq::<Thirteen>::from(1),
            Zq::<Thirteen>::from(7),
            Zq::<Thirteen>::from(9),
            Zq::<Thirteen>::from(10)
        ]
    );
}

#[test]
fn test_mul_overflow() {
    assert_eq!(
        Zq::<Thirteen>::from(12) * Zq::<Thirteen>::from(2),
        Zq::<Thirteen>::from(11)
    );
}

#[test]
fn test_add_overflow() {
    assert_eq!(
        Zq::<Thirteen>::from(12) + Zq::<Thirteen>::from(2),
        Zq::<Thirteen>::from(1)
    );
}
#[test]
fn test_sub_overflow() {
    assert_eq!(
        Zq::<Thirteen>::from(2) - Zq::<Thirteen>::from(7),
        Zq::<Thirteen>::from(8)
    );
}

proptest! {
    #[test]
    fn test_mul_is_associative(a in arb_zq_p256(), b in arb_zq_p256(), c in arb_zq_p256()) {
        prop_assert_eq!(a * (b * c), (a * b) * c);
    }

    #[test]
    fn test_mul_is_commutative(a in arb_zq_p256(), b in arb_zq_p256()) {
        prop_assert_eq!(a * b, b * a);
    }

    #[test]
    fn test_mul_is_distributive(a in arb_zq_p256(), b in arb_zq_p256(), c in arb_zq_p256()) {
        prop_assert_eq!(a * (b + c), (a * b) + (a * c));
    }

    #[test]
    fn test_mul_identity(a in arb_zq_p256()) {
        prop_assert_eq!(a * Zq::<P256>::one(), a);
    }

    #[test]
    fn test_mul_zero(a in arb_zq_p256()) {
        prop_assert_eq!(a * Zq::<P256>::zero(), Zq::<P256>::zero());
    }

    #[test]
    fn batch_inversion_equiv_manual(zqs in proptest::array::uniform10(arb_zq_p256())) {
        prop_assume!(zqs.iter().all(|z| !z.is_zero()));
        let batched = Zq::batch_invert(&zqs);
        let manual: [_; 10] = std::array::from_fn(|i| zqs[i].inv());
        prop_assert_eq!(&batched, &manual);
    }

    #[test]
    fn inv_is_inverse(a in arb_zq_p256()) {
        prop_assume!(!a.is_zero());
        prop_assert!((a * a.inv()).is_one());
    }

    #[test]
    fn inv_is_involution(a in arb_zq_thirteen()) {
        prop_assume!(!a.is_zero());
        prop_assert_eq!(a.inv().inv(), a);
    }

    #[test]
    fn square_roots(a in arb_zq_p256()) {
        prop_assume!(a.legendre_symbol() == 1);
        let Some((y0, y1)) = a.square_roots() else {
            unreachable!()
        };
        prop_assert_eq!(y0.square(), a);
        prop_assert_eq!(y1.square(), a);
    }

    #[test]
    fn test_pow(a in arb_zq_p256(), b in 0..10u64) {
        let res1 = a.pow(b);
        let mut res2 = Zq::<P256>::one();
        for _ in 0..b {
            res2 *= a;
        }
        prop_assert_eq!(res1, res2);
    }

    #[test]
    fn test_pow_zero(a in arb_zq_p256()) {
        prop_assert_eq!(a.pow(0), Zq::<P256>::one());
    }

    #[test]
    fn test_pow_one(a in arb_zq_p256()) {
        prop_assert_eq!(a.pow(1), a);
    }

    #[test]
    fn test_last_pow(a in arb_zq_p256(), b in 1..u64::MAX) {
        let res1 = a.pow(b);
        let res2 = a.pow(b - 1);
        prop_assert_eq!(res1, res2 * a);
    }

    #[test]
    fn test_neg(a in arb_zq_p256()) {
        prop_assert_eq!(a + (-a), Zq::<P256>::zero());
    }
}

#[test]
#[should_panic = "0 has no modular inverse"]
fn inv_panics_on_zero() {
    Zq::<P256>::zero().inv();
}

#[test]
fn test_inv_one() {
    // Inverse of 1 should be 1
    let one = Zq::<Thirteen>::one();
    assert_eq!(one.inv(), one);
}

#[test]
fn test_inv_division() {
    // Test that division uses inversion correctly
    let a = Zq::<Thirteen>::from(6);
    let b = Zq::<Thirteen>::from(2);
    let c = a / b;
    assert_eq!(c * b, a);
    // 6 / 2 = 6 * 7 = 42 = 3 (mod 13)
    assert_eq!(c, Zq::<Thirteen>::from(3));
}

#[test]
fn test_legendre_symbol_mod_13() {
    /*
    Python code to generate the legendre symbols:
    elts = list(range(13))
    MOD = 13
    legendre_exp = int((MOD - 1)/2)
    legendre_symbols = list(map(lambda x: -1 if x==MOD-1 else x, map(lambda x: (x**legendre_exp) % MOD, elts)))
    list(zip(elts, legendre_symbols))
    [(0, 0), (1, 1), (2, -1), (3, 1), (4, 1), (5, -1), (6, -1), (7, -1), (8, -1), (9, 1), (10, 1), (11, -1), (12, 1)]
     */
    type F13 = Zq<Thirteen>;

    // Zero
    assert_eq!(F13::from(0).legendre_symbol(), 0);

    // Quadratic residues: {1, 3, 4, 9, 10, 12}
    // (1²=1, 2²=4, 3²=9, 4²≡3, 5²≡12, 6²≡10)
    for qr in [1, 3, 4, 9, 10, 12] {
        assert_eq!(F13::from(qr).legendre_symbol(), 1);
    }

    // Non-residues: {2, 5, 6, 7, 8, 11}
    for nqr in [2, 5, 6, 7, 8, 11] {
        assert_eq!(F13::from(nqr).legendre_symbol(), -1);
    }
}
