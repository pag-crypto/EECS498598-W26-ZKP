// Slices

fn first_word(s: &String) -> usize {
    let bytes = s.as_bytes();

    for (i, &item) in bytes.iter().enumerate() {
        if item == b' ' {
            return i;
        }
    }

    s.len()
}

fn prints_first_word(mut s: String) {
    let first_word_index = first_word(&s);
    println!("first word ends at {first_word_index}");
    s.clear();
    // first_word_index really should be invalidated here somehow
}

fn string_slices(s: String) {
    let slice1 = &s[..3];
    let slice2 = &s[3..5];
    let slice3 = &s[5..10];
    //s.clear();
    println!("{}", slice1.len());
    println!("{}", slice2.len());
    println!("{}", slice3.len());
}

fn first_word_redux(s: &String) -> &str {
    let bytes = s.as_bytes();

    for (i, &item) in bytes.iter().enumerate() {
        if item == b' ' {
            return &s[0..i];
        }
    }

    &s[..]
}

fn prints_first_word2(mut s: String) {
    let word = first_word_redux(&s);
    //s.clear();
    println!("first word {word}");
}

fn other_slices(v: Vec<i32>) {
    let slice: &[i32] = &v[0..3];
    // &str &[i32]
    // str [i32]
    // String Vec ([i32; N])

    let array: [i32; 6] = [0, 1, 2, 3, 4, 5];
    let slice = &array[0..2];
}

// &String
// &Vec<i32> &[i32]
fn first_word_final(s: &str) -> &str {
    let bytes = s.as_bytes();

    for (i, &item) in bytes.iter().enumerate() {
        if item == b' ' {
            return &s[0..i];
        }
    }

    &s[..]
}

fn call_first_word(s: String) {
    first_word_final(&s);
}
