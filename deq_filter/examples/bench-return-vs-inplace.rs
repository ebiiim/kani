use std::env;
use std::time;

// Tested in Core i7-4790:
//
// vector size: 1024 (4096 bytes)
// (return, in-place): (4966, 454)
// 10.938X
//
// vector size: 2048 (8192 bytes)
// (return, in-place): (11067, 775)
// 14.280X
//
// vector size: 8192 (32768 bytes)
// (return, in-place): (66678, 4028)
// 16.554X
//
// vector size: 1048576 (4194304 bytes)
// (return, in-place): (5658557, 611935)
// 9.247X
//
// vector size: 1073741824 (4294967296 bytes)
// (return, in-place): (5711876735, 1675055027)
// 3.410X

fn help() {
    println!(
        "Usage: cargo run --release --example bench-return-vs-inplace N\n    N   vector.len()=2^N"
    );
    std::process::exit(1);
}

fn main() {
    let mut n: usize = 10;

    let args: Vec<String> = env::args().collect();
    match args.len() {
        1 => help(),
        2 => match args[1].parse::<usize>() {
            Ok(b) => {
                n = b;
            }
            Err(_) => {
                help();
            }
        },
        _ => help(),
    }

    let a1: Vec<f32> = vec![0.12345; 1 << n];
    let start_a = time::Instant::now();
    let a2: Vec<f32> = a1.iter().map(|x| x * 1.12345).collect();
    let a3: Vec<f32> = a2.iter().map(|x| x * 2.12345).collect();
    let a4: Vec<f32> = a3.iter().map(|x| x * 3.12345).collect();
    let a5: Vec<f32> = a4.iter().map(|x| x * 4.12345).collect();
    let end_a = start_a.elapsed();

    let mut b1: Vec<f32> = vec![0.12345; 1 << n];
    let start_b = time::Instant::now();
    b1.iter_mut().for_each(|x| *x *= 1.12345);
    b1.iter_mut().for_each(|x| *x *= 2.12345);
    b1.iter_mut().for_each(|x| *x *= 3.12345);
    b1.iter_mut().for_each(|x| *x *= 4.12345);
    let end_b = start_b.elapsed();

    println!(
        "vector size: {} ({} bytes)\n(return, in-place): ({}, {})\n{:.3}X",
        1 << n,
        std::mem::size_of_val(&*b1),
        end_a.as_nanos(),
        end_b.as_nanos(),
        end_a.as_nanos() as f64 / end_b.as_nanos() as f64,
    );
    assert_eq!(a5[0], b1[0]);
}
