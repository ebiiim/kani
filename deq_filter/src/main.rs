use deq_filter as f;
use deq_filter::Filter;
use deq_filter::{BQFParam, BQFType, BiquadFilter};
use deq_filter::{Volume, VolumeCurve};

fn run_dump_coeffs() {
    let fs = 48000.0;
    let v: Vec<BiquadFilter> = vec![
        BiquadFilter::new(BQFType::HighPass, fs, 250.0, 0.0, BQFParam::Q(0.707)),
        BiquadFilter::new(BQFType::LowPass, fs, 8000.0, 0.0, BQFParam::Q(0.707)),
        // BiquadFilter::new(BQFType::PeakingEQ, fs, 880.0, 9.0, BQFParam::BW(1.0)),
    ];
    print!("{}", f::dump_coeffs(&v));
}

fn run_dump_ir() {
    let fs = 48000.0;
    let n = f::nextpow2(fs / 20.0);
    let mut v: f::VecFilters = vec![
        BiquadFilter::newb(BQFType::HighPass, fs, 250.0, 0.0, BQFParam::Q(0.707)),
        BiquadFilter::newb(BQFType::LowPass, fs, 8000.0, 0.0, BQFParam::Q(0.707)),
        // BiquadFilter::newb(BQFType::PeakingEQ, fs, 880.0, 9.0, BQFParam::BW(1.0)),
        Volume::newb(VolumeCurve::Gain, -6.0),
    ];
    println!("{}", f::dump_ir(&mut v, n));
}

fn run_dump_ir_vocal_remover() {
    const FS: f32 = 48000.0;
    let len = f::nextpow2(FS);
    let a = f::generate_impulse(len);

    // find values that can obtain flat frequency response
    let q = BQFParam::Q(0.707);
    let r1 = 1.414;
    // let q = BQFParam::Q(1.414);
    // let r1 = 0.882; // better than 0.707
    let r2 = 1.0 / r1;
    let f1 = 200.0;
    let f11 = f1 * r1;
    let f12 = f1 * r2;
    let f2 = 5400.0;
    let f21 = f2 * r1;
    let f22 = f2 * r2;

    let l = BiquadFilter::newb(BQFType::LowPass, FS, f11, 0.0, q).apply(&a);
    let h = BiquadFilter::newb(BQFType::HighPass, FS, f22, 0.0, q).apply(&a);

    let x = BiquadFilter::newb(BQFType::HighPass, FS, f12, 0.0, q).apply(&a);
    let x = BiquadFilter::newb(BQFType::LowPass, FS, f21, 0.0, q).apply(&x);
    let x = Volume::newb(VolumeCurve::Gain, 0.0).apply(&x);

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
    // run_dump_coeffs();
    // run_dump_ir();
    run_dump_ir_vocal_remover();
}
