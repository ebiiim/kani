use deq_filter as f;
use deq_filter::Filter;
use deq_filter::{BQFParam, BQFType, BiquadFilter};

fn test_run_dump_coeffs() {
    let fs = 48000.0;
    let v: Vec<BiquadFilter> = vec![
        BiquadFilter::new(BQFType::PeakingEQ, fs, 440.0, -6.0, BQFParam::Q(2.828)),
        BiquadFilter::new(BQFType::PeakingEQ, fs, 440.0, -6.0, BQFParam::Q(2.828)),
        BiquadFilter::new(BQFType::LowPass, fs, 8000.0, 0.0, BQFParam::Q(0.707)),
    ];
    print!("{}", f::dump_coeffs(v));
}

fn test_run_dump_ir() {
    let fs = 48000.0;
    let n = f::nextpow2(fs / 20.0);
    let v: Vec<Box<dyn Filter>> = vec![
        BiquadFilter::newb(BQFType::PeakingEQ, fs, 440.0, -6.0, BQFParam::Q(2.828)),
        BiquadFilter::newb(BQFType::PeakingEQ, fs, 440.0, -6.0, BQFParam::Q(2.828)),
        BiquadFilter::newb(BQFType::LowPass, fs, 8000.0, 0.0, BQFParam::Q(0.707)),
    ];
    println!("{}", f::dump_ir(v, n));
}

fn main() {
    test_run_dump_coeffs();
    test_run_dump_ir();
}
