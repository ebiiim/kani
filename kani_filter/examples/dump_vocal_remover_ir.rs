use kani_filter as f;
use kani_filter::*;

fn run_dump_vocal_remover_ir() {
    let fs = 48000;
    let len = f::nextpow2(fs);
    let a = f::generate_impulse(len);

    // find values that can obtain flat frequency response

    // let q = BQFParam::Q(0.707);
    // let r1 = 1.414;
    let q = BQFParam::Q(1.414);
    let r1 = 0.882; // better than 0.707
    let r2 = 1.0 / r1;

    let f1 = 200.0;
    let f11 = f1 * r1;
    let f12 = f1 * r2;

    let f2 = 5400.0;
    let f21 = f2 * r1;
    let f22 = f2 * r2;

    let l = BiquadFilter::newb(BQFType::LowPass, fs, f11 as f::Hz, 0.0, q).apply(&a);
    let h = BiquadFilter::newb(BQFType::HighPass, fs, f22 as f::Hz, 0.0, q).apply(&a);

    let x = BiquadFilter::newb(BQFType::HighPass, fs, f12 as f::Hz, 0.0, q).apply(&a);
    let x = BiquadFilter::newb(BQFType::LowPass, fs, f21 as f::Hz, 0.0, q).apply(&x);

    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        out.push(l[i] + x[i] + h[i]);
    }

    println!(
        "{}",
        out.iter()
            .fold(String::new(), |s, &x| s + &x.to_string() + "\n")
    );
}

fn main() {
    run_dump_vocal_remover_ir();
}
