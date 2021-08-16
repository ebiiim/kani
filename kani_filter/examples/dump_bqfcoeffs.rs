use kani_filter as f;
use kani_filter::*;

fn run_dump_bqfcoeffs() {
    let fs = 48000.0;
    let v: Vec<BiquadFilter> = vec![
        BiquadFilter::new(BQFType::HighPass, fs, 250.0, 0.0, BQFParam::Q(0.707)),
        BiquadFilter::new(BQFType::LowPass, fs, 8000.0, 0.0, BQFParam::Q(0.707)),
        BiquadFilter::new(BQFType::PeakingEQ, fs, 880.0, 9.0, BQFParam::BW(1.0)),
    ];
    print!("{}", f::dump_coeffs(&v));
}

fn main() {
    run_dump_bqfcoeffs();
}
