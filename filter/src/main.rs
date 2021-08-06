use filter::Appliable;
use filter::{BQFParam, BQFType, BiquadFilter};

fn test_run_dump_coeffs() {
    let fs = 48000.0;
    let v: Vec<BiquadFilter> = vec![
        BiquadFilter::new(BQFType::PeakingEQ, fs, 440.0, -6.0, BQFParam::Q(2.828)),
        BiquadFilter::new(BQFType::PeakingEQ, fs, 440.0, -6.0, BQFParam::Q(2.828)),
        BiquadFilter::new(BQFType::LowPass, fs, 8000.0, 0.0, BQFParam::Q(0.707)),
    ];
    print!("{}", filter::dump_coeffs(v));
}

fn test_run_dump_ir() {
    let fs = 48000.0;
    let n = filter::nextpow2(fs / 20.0);
    let v: Vec<Box<dyn Appliable>> = vec![
        BiquadFilter::newb(BQFType::PeakingEQ, fs, 440.0, -6.0, BQFParam::Q(2.828)),
        BiquadFilter::newb(BQFType::PeakingEQ, fs, 440.0, -6.0, BQFParam::Q(2.828)),
        BiquadFilter::newb(BQFType::LowPass, fs, 8000.0, 0.0, BQFParam::Q(0.707)),
    ];
    println!("{}", filter::dump_ir(v, n));
}

fn main() {
    test_run_dump_coeffs();
    test_run_dump_ir();
}
