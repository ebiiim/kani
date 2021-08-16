use std::env;
use std::time;

// Tested in Ryzen 9 5900X:
//
// N=10
// vector size: 1024 (4096 bytes)
// (return, in-place): (461, 216)
// 2.133X
//
// N=11
// vector size: 2048 (8192 bytes)
// (return, in-place): (815, 433)
// 1.882X
//
// N=13
// vector size: 8192 (32768 bytes)
// (return, in-place): (2922, 1728)
// 1.691X
//
// N=20
// vector size: 1048576 (4194304 bytes)
// (return, in-place): (2540603, 233452)
// 10.883X
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

    let loop_num = 100000;
    let mut a: Vec<f32> = vec![0.12345; 1 << n];
    let start_a = time::Instant::now();
    for _ in 0..loop_num {
        a = a.iter().map(|x| x * 1.12345 * 1.12345 * 1.12345).collect();
        a = a.iter().map(|x| x * 2.12345 * 2.12345 * 2.12345).collect();
        a = a.iter().map(|x| x * 3.12345 * 3.12345 * 3.12345).collect();
        a = a.iter().map(|x| x * 4.12345 * 4.12345 * 4.12345).collect();
    }
    let end_a = start_a.elapsed();

    let mut b: Vec<f32> = vec![0.12345; 1 << n];
    let start_b = time::Instant::now();
    for _ in 0..loop_num {
        b.iter_mut().for_each(|x| *x *= 1.12345 * 1.12345 * 1.12345);
        b.iter_mut().for_each(|x| *x *= 2.12345 * 2.12345 * 2.12345);
        b.iter_mut().for_each(|x| *x *= 3.12345 * 3.12345 * 3.12345);
        b.iter_mut().for_each(|x| *x *= 4.12345 * 4.12345 * 4.12345);
    }
    let end_b = start_b.elapsed();

    println!(
        "vector size: {} ({} bytes)\n(return, in-place): ({}, {})\n{:.3}X",
        1 << n,
        std::mem::size_of_val(&*b),
        end_a.as_nanos() / loop_num,
        end_b.as_nanos() / loop_num,
        end_a.as_nanos() as f64 / end_b.as_nanos() as f64,
    );
    assert_eq!(a[0], b[0]);
}
