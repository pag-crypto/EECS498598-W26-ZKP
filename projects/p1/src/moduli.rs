/// This module defines type-level moduli. You can feel free to add more moduli
/// here for testing purposes, but there is no need for student modifications to this file.
use sfs_bigint::U256;
/// Represents a (prime) modulus at the type level. New moduli should be constructed via
/// `declare_const_modulus!` macro.
///
/// However, one must ensure that any such modulus is in fact prime.
pub trait PrimeModulus: Send + Sync + 'static {
    /// actual value of the modulus
    const VALUE: U256;

    // (VALUE - 1) / 2
    // `const` traits are not stable in rust yet, so we can't compute (VALUE - 1) / 2 in a const
    // context.
    // the best we can do is write a function that will compute (p-1)/2 exactly once
    fn legendre_exponent() -> &'static U256;
}

#[macro_export]
macro_rules! declare_const_modulus {
    // INVARIANT: Modulus should be prime!
    ($mod_name: ident, $value_str:expr) => {
        declare_const_modulus!($mod_name, $value_str, 10);
    };
    ($mod_name:ident, $value_str:expr, $radix:expr) => {
        pub enum $mod_name {}

        impl $crate::moduli::PrimeModulus for $mod_name {
            const VALUE: sfs_bigint::U256 =
                sfs_bigint::U256::from_str_radix_const($value_str, $radix);
            fn legendre_exponent() -> &'static sfs_bigint::U256 {
                static LEGENDRE_EXPONENT: std::sync::LazyLock<sfs_bigint::U256> =
                    std::sync::LazyLock::new(|| {
                        (<$mod_name as $crate::moduli::PrimeModulus>::VALUE - 1.into()) / 2.into()
                    });
                &LEGENDRE_EXPONENT
            }
        }
    };
}

declare_const_modulus!(
    Ed25519,
    "57896044618658097711785492504343953926634992332820282019728792003956564819949"
);

// Useful for testing
declare_const_modulus!(Thirteen, "13");
//This is 3 mod 4 so use easy sqrt
declare_const_modulus!(
    P256,
    "115792089210356248762697446949407573530086143415290314195533631308867097853951"
);
//This is equal to one mod 4, so tonelli-shanks
declare_const_modulus!(
    P256CurveOrder,
    "115792089210356248762697446949407573529996955224135760342422259061068512044369"
);
