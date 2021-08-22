use kani_filter::*;

fn dump_coeffs(v: &[BiquadFilter]) -> String {
    v.iter()
        .fold(String::new(), |s, x| format!("{}{}", s, x.dump_coeff()))
}

fn run_dump_bqfcoeffs() {
    let fs = 48000;
    let v: Vec<BiquadFilter> = vec![
        BiquadFilter::new(BQFType::HighPass, fs, 250, 0.0, BQFParam::Q(0.707)),
        BiquadFilter::new(BQFType::LowPass, fs, 8000, 0.0, BQFParam::Q(0.707)),
        BiquadFilter::new(BQFType::PeakingEQ, fs, 880, 9.0, BQFParam::BW(1.0)),
    ];
    print!("{}", dump_coeffs(&v));
}

fn main() {
    run_dump_bqfcoeffs();
}
