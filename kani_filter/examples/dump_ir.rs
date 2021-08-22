use kani_filter as f;
use kani_filter::*;

fn run_dump_ir() {
    let fs = 48000;
    let n = f::nextpow2(fs / 20);
    let mut v: f::VecFilters = vec![
        BiquadFilter::newb(BQFType::HighPass, fs, 250, 0.0, BQFParam::Q(0.707)),
        BiquadFilter::newb(BQFType::LowPass, fs, 8000, 0.0, BQFParam::Q(0.707)),
        BiquadFilter::newb(BQFType::PeakingEQ, fs, 880, 9.0, BQFParam::BW(1.0)),
        Volume::newb(VolumeCurve::Gain, -6.0),
    ];
    println!("{}", f::dump_ir(&mut v, n));
}

fn main() {
    run_dump_ir();
}
