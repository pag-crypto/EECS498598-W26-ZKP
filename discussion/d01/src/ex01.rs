fn immutability() {
    let x = 0;
    println!("{x}");
    x = 2;
    println!("{x}");
}

fn shadowing() {
    let x = 0;
    println!("{x}");
    let x = 2;
    println!("{x}");
    let x = "hello";
    println!("{x}");
}

fn shadowing2() {
    let mut my_str = String::from("hello, ");
    my_str += "world";
    my_str += "!";
    let my_str = my_str;
}

fn takes_mut(mut x: i32) -> i32 {
    x += 1;
    x
}

fn calls_mut() {
    let x = 1;
    println!("{}", takes_mut(x));
}

fn give_me_i32(x: i32) {}

fn takes_i64(x: i64) {
    give_me_i32(x as i32)
}

const MY_CONST: i32 = 12;

static MY_STATIC: i32 = 12;
static mut MY_EVIL_STATIC: i32 = 12;

fn expressions(y: i32, z: i32) -> i32 {
    let x = if y == z { 2 } else { 3 };
    if y == 0 {
        let w = x * 3;
        w
    } else if y == 10 {
        z
    } else {
        x
    }
}

fn sum_vector() -> i32 {
    let v = vec![1, 2, 3, 4, 5];

    let mut i = 0;
    let mut sum = 0;
    'label: loop {
        if i >= v.len() {
            break;
        }
        sum += v[i];
    }

    sum = 0;
    i = 0;
    while i < v.len() {
        sum += v[i];
    }

    sum = 0;
    for elem in &v {
        sum += elem;
    }

    v.iter().sum()
}
