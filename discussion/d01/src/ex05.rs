enum StringOrInt {
    Str(String),
    Int(i32),
}
enum ECPoint {
    Point { x: u128, y: u128 },
    Inf,
}

fn ecpoint_example() {
    let point = ECPoint::Point { x: 10, y: 12 };
    let inf = ECPoint::Inf;
}

fn print_ec_point(point: &ECPoint) {
    match point {
        ECPoint::Inf => println!("inf"),
        ECPoint::Point { x, y } => println!("{x}, {y}"),
        ECPoint::Point { x, y } => println!("{x}, {y}"),
    }
}

fn safe_div(x: i32, y: i32) -> Option<i32> {
    if y == 0 { None } else { Some(x / y) }
}

fn div_add_one(a: i32, b: i32) -> Option<i32> {
    //safe_div(a, b) + 1
}

fn other_pattern_matching(slice: &[u8]) {}

struct Point {
    x: i32,
    y: i32,
}

fn let_pattern_matching(p: Point) {}

// let else + if let
