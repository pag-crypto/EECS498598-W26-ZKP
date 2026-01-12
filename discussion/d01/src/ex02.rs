fn ownership1() {
    let s = String::from("hello");
    let s2 = s.clone();
    println!("{s}");
    println!("{s2}");
}

fn ownership_copy() {
    let i = 12;
    let i2 = i;
    println!("{i}");
    println!("{i2}");
}

fn takes_ownership(s: &String) {
    println!("{s}");
}

fn makes_copy(i: i32) {
    println!("{i}");
}

// REFERENCES

fn prints_length() {
    let mut s = String::from("hello, world");
    println!("length {s} is {}", string_length(&s));
    add_exclamation_point(&mut s);
    println!("length {s} is {}", string_length(&s));
}

fn string_length(s: &String) -> usize {
    s.len()
}

fn add_exclamation_point(s: &mut String) {
    *s += "!";
}

fn aliasing_xor_mutability(s: String) {
    let mut s = String::from("hello");
    let sref1 = &s;
    let sref2 = &s;
    let mutref1 = &mut s;
    let mutref2 = &mut s;
}

fn reference_invalidation(mut v: Vec<i32>) -> Vec<i32> {
    let x = &v[0];

    v.push(*x);
    println!("{x}");
    v
}

fn returns_ref(s: String) -> &'static String {
    &s
}
