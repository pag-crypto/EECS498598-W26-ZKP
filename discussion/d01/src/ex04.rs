struct Point {
    x: i32,
    y: i32,
}

fn point_example() {
    let p = Point { x: 10, y: 12 };
    let p2 = Point { x: 5, ..p };
    assert_eq!(p.y, p2.y);
}

struct Point2(i32, i32);

fn point_example2() {
    let p = Point2(10, 12);
    println!("({}, {})", p.0, p.1);
}

struct Point0D;

fn point_example3() {
    let point = Point0D;
}

impl Point {
    fn new(x: i32, y: i32) -> Self {
        Point { x, y }
    }
    fn add(&self, other: &Self) -> Self {
        Point {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

fn print_point_sum(points: Vec<Point>) -> Point {
    let mut sum = Point::new(0, 0);
    for point in points {
        sum = sum.add(&point);
    }
    sum
}
