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
    let mut v: Vec<Box<dyn Filter>> = vec![
        BiquadFilter::newb(BQFType::HighPass, fs, 250.0, 0.0, BQFParam::Q(0.707)),
        BiquadFilter::newb(BQFType::LowPass, fs, 8000.0, 0.0, BQFParam::Q(0.707)),
        // BiquadFilter::newb(BQFType::PeakingEQ, fs, 880.0, 9.0, BQFParam::BW(1.0)),
        Volume::newb(VolumeCurve::Gain, -6.0),
    ];
    println!("{}", f::dump_ir(&mut v, n));
}

fn main() {
    // run_dump_coeffs();
    run_dump_ir();
}
