// GOALS: implement
// constructor, copy/cloning, equality, printing, and arithmetic operations (add, sub, neg, scalar
// mult)
use std::ops::{Add, Mul};
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct GenericPoint<T> {
    x: T,
    y: T,
}

impl<T> GenericPoint<T> {
    pub fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

impl<T> Add for GenericPoint<T>
where
    T: Add<T>,
{
    type Output = GenericPoint<T::Output>;

    fn add(self, rhs: Self) -> Self::Output {
        GenericPoint {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl<T: Mul<Output = T> + Clone> Mul<T> for GenericPoint<T> {
    type Output = GenericPoint<T>;
    fn mul(self, rhs: T) -> Self::Output {
        GenericPoint {
            x: self.x.mul(rhs.clone()),
            y: self.y.mul(rhs),
        }
    }
}

fn test() {
    let p = GenericPoint::new(1, 2);
    let p = GenericPoint::new(3, 4);
}
