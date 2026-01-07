use num_traits::Zero;
use p1::curve::P256Point;
use p1::moduli::P256CurveOrder;
use p1::zq::Zq;
use proptest::prelude::*;
use std::str::FromStr;

// Strategy function for generating arbitrary P256Point values
fn arb_p256_point() -> BoxedStrategy<P256Point> {
    any::<usize>()
        .prop_map(|x| P256Point::get_generator_from_seed(x))
        .boxed()
}

#[test]
fn test_is_not_on_curve() {
    let P256Point::Point { x, y, .. } = P256Point::GENERATOR else {
        unreachable!()
    };
    assert!(P256Point::is_on_curve(&x, &y));
    //This point can't be on the curve because rhs(x) is not 1.
    assert!(!P256Point::is_on_curve(&x, &1.into()));
}

proptest! {
    #[test]
    fn test_add_associativity(p1 in arb_p256_point(), p2 in arb_p256_point(), p3 in arb_p256_point()) {
        prop_assert_eq!(p1 + (p2 + p3), (p1 + p2) + p3);
    }

    #[test]
    fn test_sub_is_add_inverse(p in arb_p256_point()) {
        prop_assert_eq!(p - p, P256Point::zero());
    }

    #[test]
    fn test_sub(p in arb_p256_point(), q in arb_p256_point()) {
        prop_assert_eq!(p - q, p + (-q));
    }

    #[test]
    fn test_add_point_at_infinity(p in arb_p256_point()) {
        let g2 = p.clone() + P256Point::zero();
        prop_assert_eq!(g2, p);
    }

    #[test]
    fn test_commutativity(p1 in arb_p256_point(), p2 in arb_p256_point()) {
        prop_assert_eq!(p1 + p2, p2 + p1);
    }

    #[test]
    fn test_stays_on_curve(p1 in arb_p256_point(), p2 in arb_p256_point()) {
        let result = p1 + p2;
        if let P256Point::Point { x, y, .. } = result {
            prop_assert!(P256Point::is_on_curve(&x, &y));
        } else {
            prop_assert_eq!(p1, -&p2);
            prop_assert_eq!(result, P256Point::Inf);
        };
    }

    #[test]
    fn test_point_plus_neg_is_inf(p in arb_p256_point()) {
        let result = -&p + p;
        prop_assert_eq!(result, P256Point::Inf);
    }
}

#[test]
fn test_mul() {
    //get the generator
    let g = P256Point::GENERATOR;
    //add it to itself three times
    let g2 = g + g;
    let g3 = g2 + g;
    //compute 3*G using mul
    let g3mul = g * 3.into();
    //assert they're the same
    assert_eq!(g3, g3mul);
    let g0 = g * 0.into();
    assert!(g0.is_zero());
}

#[test]
fn test_msm() {
    let g = P256Point::GENERATOR;
    let bases = vec![g.clone(), g.clone(), g.clone()];
    let scalars = vec![1, 1, 0]
        .iter()
        .map(|x| Zq::<P256CurveOrder>::from(*x))
        .collect::<Vec<_>>();
    let result = P256Point::msm(&scalars, &bases);
    let g2 = g + g;
    assert_eq!(result, g2);
}

#[test]
fn test_is_on_curve() {
    // Access the generator constants through string parsing
    let g_str = "48439561293906451759052585252797914202762949526041747995844080717082404635286,36134250956749795798585127919587881956611106672985015071877198253568414405109";
    let g = P256Point::from_str(g_str).expect("generator is on curve");
    if let P256Point::Point { x, y, .. } = g {
        assert!(P256Point::is_on_curve(&x, &y));
    }
}

#[test]
fn test_generator() {
    // Test that we can get the generator using the static method
    let g = P256Point::GENERATOR;

    // Verify it's the same as creating it from the string constants
    let g_str = "48439561293906451759052585252797914202762949526041747995844080717082404635286,36134250956749795798585127919587881956611106672985015071877198253568414405109";
    let g_from_str = P256Point::from_str(g_str).unwrap();
    assert_eq!(g, g_from_str);

    // We can also use it multiple times
    let g2 = P256Point::GENERATOR;
    assert_eq!(g, g2);
    let P256Point::Point { x, y, .. } = g else {
        panic!("g is inf point");
    };
    println!("coordinates: {}, {}", &x, &y);
}
