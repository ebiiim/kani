use std::collections::VecDeque;

// TODO: consider in-place updates

/// [-32768, 32767] -> [-1.0, 1.0]
pub fn i16_to_f32(a: Vec<i16>) -> Vec<f32> {
    a.into_iter()
        .map(|x| {
            if x < 0 {
                x as f32 / 0x8000 as f32
            } else {
                x as f32 / 0x7fff as f32
            }
        })
        .collect()
}

/// [-1.0, 1.0] -> [-32768, 32767]
pub fn f32_to_i16(a: Vec<f32>) -> Vec<i16> {
    a.into_iter()
        .map(|x| {
            if x < 0.0 {
                (x * 0x8000 as f32) as i16
            } else {
                (x * 0x7fff as f32) as i16
            }
        })
        .collect()
}

#[test]
#[allow(clippy::float_cmp)]
fn test_i16_to_f32() {
    let want: Vec<f32> = vec![
        0.0,
        3.051851E-5,
        6.103702E-5,
        9.155553E-5,
        -3.0517578E-5,
        -6.1035156E-5,
        -9.1552734E-5,
        1.0,
        -1.0,
    ];
    let buf = vec![0, 1, 2, 3, -1, -2, -3, std::i16::MAX, std::i16::MIN];
    let got = i16_to_f32(buf);
    let ok = got.iter().zip(&want).filter(|&(a, b)| a != b).count() == 0;
    assert!(ok);
}

#[test]
#[allow(clippy::float_cmp)]
fn test_f32_to_i16() {
    let want: Vec<i16> = vec![
        0,
        1,
        2,
        3,
        -1,
        -2,
        -3,
        std::i16::MAX,
        std::i16::MIN,
        std::i16::MAX,
        std::i16::MIN,
        std::i16::MAX,
        std::i16::MIN,
    ];
    let buf = vec![
        0.0,
        3.051851E-5,
        6.103702E-5,
        9.155553E-5,
        -3.0517578E-5,
        -6.1035156E-5,
        -9.1552734E-5,
        1.0,
        -1.0,
        // round to [-1.0, 1.0]
        1.0000001,
        -1.0000001,
        123.456,
        -123.456,
    ];
    let got = f32_to_i16(buf);
    let ok = got.iter().zip(&want).filter(|&(a, b)| a != b).count() == 0;
    assert!(ok);
}

pub fn from_interleaved(a: Vec<f32>) -> (Vec<f32>, Vec<f32>) {
    let mut l: Vec<f32> = Vec::with_capacity(a.len() / 2);
    let mut r: Vec<f32> = Vec::with_capacity(a.len() / 2);
    for (i, s) in a.into_iter().enumerate() {
        if i % 2 == 0 {
            l.push(s);
        } else {
            r.push(s);
        }
    }
    (l, r)
}

/// Interleaves two vectors.
///
/// # Panics
///
/// Panics if `l.len()` != `r.len()`.
pub fn to_interleaved(l: Vec<f32>, r: Vec<f32>) -> Vec<f32> {
    if l.len() != r.len() {
        panic!(
            "vectors must have same length but l={} r={}",
            l.len(),
            r.len()
        )
    }
    let mut a: Vec<f32> = Vec::with_capacity(l.len() * 2);
    for (lv, rv) in l.into_iter().zip(r) {
        a.push(lv);
        a.push(rv);
    }
    a
}

#[test]
#[allow(clippy::float_cmp)]
fn test_from_interleaved() {
    let want_l: Vec<f32> = vec![0.1, 0.3, 0.5, 0.7];
    let want_r: Vec<f32> = vec![0.2, 0.4, 0.6, 0.8];
    let buf = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let (got_l, got_r) = from_interleaved(buf);
    let ok_l = got_l.iter().zip(&want_l).filter(|&(a, b)| a != b).count() == 0;
    let ok_r = got_r.iter().zip(&want_r).filter(|&(a, b)| a != b).count() == 0;
    assert!(ok_l && ok_r);
}

#[test]
#[allow(clippy::float_cmp)]
fn test_to_interleaved() {
    let want = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let buf_l: Vec<f32> = vec![0.1, 0.3, 0.5, 0.7];
    let buf_r: Vec<f32> = vec![0.2, 0.4, 0.6, 0.8];
    let got = to_interleaved(buf_l, buf_r);
    let ok = got.iter().zip(&want).filter(|&(a, b)| a != b).count() == 0;
    assert!(ok);
}

#[test]
#[should_panic]
fn test_to_interleaved_invalid_vec() {
    let buf_l: Vec<f32> = vec![0.1, 0.3, 0.5, 0.7];
    let buf_r: Vec<f32> = vec![0.2, 0.4, 0.6];
    to_interleaved(buf_l, buf_r);
}

pub trait Filter {
    fn apply(&mut self, samples: Vec<f32>) -> Vec<f32>;
}

#[derive(Debug)]
pub struct Delay {
    tapnum: usize,
    buf: VecDeque<f32>,
}

impl Delay {
    const MAX_TAPNUM: usize = 1 << 30 >> 2; // 1GiB f32 array
    /// Initializes Delay with given duration in millisecond and sampling rate.
    ///
    /// # Panics
    ///
    /// Panics if it needs more than 1GiB mem.
    /// See also: Delay::with_tapnum
    pub fn new(time_ms: usize, sample_rate: usize) -> Self {
        log::debug!("delay time_ms={} sample_rate={}", time_ms, sample_rate);
        Self::with_tapnum(Self::calc_tapnum(time_ms, sample_rate))
    }
    fn calc_tapnum(time_ms: usize, sample_rate: usize) -> usize {
        time_ms * sample_rate / 1000
    }
    /// Initializes Delay with given taps.
    ///
    /// # Panics
    ///
    /// Panics if tapnum > (2^28) as it needs more than 1GiB mem.
    /// FYI: tapnum=2^28 means >699000ms delay at 384kHz :D
    pub fn with_tapnum(tapnum: usize) -> Self {
        if tapnum > Self::MAX_TAPNUM {
            panic!("too long duration")
        }
        log::debug!("delay tapnum={}", tapnum);
        let buf: Vec<f32> = vec![0.0; tapnum];
        let buf = VecDeque::from(buf);
        Self { tapnum, buf }
    }
    pub fn newb(time_ms: usize, sample_rate: usize) -> Box<Self> {
        Box::new(Self::new(time_ms, sample_rate))
    }
}

impl Filter for Delay {
    fn apply(&mut self, samples: Vec<f32>) -> Vec<f32> {
        if self.tapnum == 0 {
            return samples; // do nothing
        }
        let mut y = Vec::with_capacity(samples.len());
        for x in samples.into_iter() {
            y.push(self.buf.pop_back().unwrap()); // already initialized in constructor
            self.buf.push_front(x);
        }
        y
    }
}

#[test]
#[allow(clippy::float_cmp)]
fn test_delay() {
    let want: Vec<f32> = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
        0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23,
        0.24,
        // 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, // Note: the rest is still in the buffer
    ];
    let mut buf = vec![
        0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15,
        0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30,
    ];
    buf = Delay::with_tapnum(6).apply(buf);
    let ok = buf.iter().zip(&want).filter(|&(a, b)| a != b).count() == 0;
    assert!(ok);
}

#[test]
#[allow(clippy::float_cmp)]
fn test_delay_zero() {
    let want: Vec<f32> = vec![0.01, 0.02, 0.03, 0.04];
    let mut buf = vec![0.01, 0.02, 0.03, 0.04];
    buf = Delay::with_tapnum(0).apply(buf);
    let ok = buf.iter().zip(&want).filter(|&(a, b)| a != b).count() == 0;
    assert!(ok);
}

#[test]
#[should_panic]
fn test_delay_too_long() {
    Delay::with_tapnum(Delay::MAX_TAPNUM + 1).apply(vec![0.01, 0.02, 0.03, 0.04]);
}

#[test]
#[ignore]
fn check_delay_mem() {
    // this test needs more than 4GiB RAM (1GiB for x, Delay.buf, y, respectively, and 1GiB for somewhere else ?_?)
    // `cargo test -- --ignored` to run
    let tapsize = 1 << 30 >> 2;
    let x = vec![0.123; tapsize];
    let _y = Delay::with_tapnum(tapsize).apply(x);
    // 4GiB -> 1GiB; only y remains
    use std::{thread, time};
    let t = time::Duration::from_millis(5000);
    thread::sleep(t);
}

#[derive(Debug)]
pub struct Volume {
    curve: VolumeCurve,
    val: f32,
    ratio: f32,
}

#[derive(Debug)]
pub enum VolumeCurve {
    Linear,
    Gain,
}

impl Volume {
    pub fn new(curve: VolumeCurve, val: f32) -> Self {
        let ratio = match curve {
            VolumeCurve::Linear => val,
            VolumeCurve::Gain => 10.0f32.powf(val / 20.0),
        };
        log::debug!("volume {:?}({})=>{}", curve, val, ratio);
        Self { curve, val, ratio }
    }
    pub fn newb(curve: VolumeCurve, val: f32) -> Box<Self> {
        Box::new(Self::new(curve, val))
    }
}

impl Filter for Volume {
    fn apply(&mut self, samples: Vec<f32>) -> Vec<f32> {
        samples.into_iter().map(|x| x * self.ratio as f32).collect()
    }
}

#[test]
#[allow(clippy::float_cmp)]
fn test_volume_linear() {
    let want: Vec<f32> = vec![0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5];
    let mut buf = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    buf = Volume::new(VolumeCurve::Linear, 0.5).apply(buf);
    assert!(buf.iter().zip(&want).filter(|&(a, b)| a != b).count() == 0)
}

#[test]
#[allow(clippy::float_cmp)]
fn test_volume_gain() {
    let mut buf = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let g_n6 = 0.5011872f32;
    let want: Vec<f32> = buf.clone().iter().map(|x| x * g_n6).collect();
    buf = Volume::new(VolumeCurve::Gain, -6.0).apply(buf);
    assert!(buf.iter().zip(&want).filter(|&(a, b)| a != b).count() == 0)
}

// Biquad Filter based on [RBJ Cookbook](https://webaudio.github.io/Audio-EQ-Cookbook/Audio-EQ-Cookbook.txt)

#[derive(Debug)]
pub enum BQFType {
    LowPass,
    HighPass,
    // constant 0 dB peak gain
    BandPass,
    Notch,
    AudioPeak,
    PeakingEQ,
    LowShelf,
    HighShelf,
}

#[derive(PartialEq, Debug)]
pub enum BQFParam {
    // Q factor
    Q(f32),
    // Bandwidth (Octave)
    BW(f32),
    // Slope
    S(f32),
}

#[derive(Debug)]
struct BQFCoeff {
    b0: f32,
    b1: f32,
    b2: f32,
    a0: f32,
    a1: f32,
    a2: f32,
    b0_div_a0: f32,
    b1_div_a0: f32,
    b2_div_a0: f32,
    a1_div_a0: f32,
    a2_div_a0: f32,
}

impl BQFCoeff {
    pub fn new(filter_type: &BQFType, rate: f32, f0: f32, gain: f32, param: &BQFParam) -> BQFCoeff {
        let a = 10.0f32.powf(gain / 40.0);
        let w0 = 2.0 * std::f32::consts::PI * f0 / rate;
        let w0_cos = w0.cos();
        let w0_sin = w0.sin();
        let alpha = match param {
            BQFParam::Q(q) => w0_sin / (2.0 * q),
            BQFParam::BW(bw) => {
                // I'm not so sure about this
                // w0_sin * (2.0f32.log10() / 2.0 * bw * w0 / w0_sin).sinh()
                // So I'd like to convert Bandwidth (octave) to Q based on formula [here](http://www.sengpielaudio.com/calculator-bandwidth.htm)
                let bw2q = |bw: f32| (2.0f32.powf(bw)).sqrt() / (2.0f32.powf(bw) - 1.0);
                w0_sin / (2.0 * bw2q(*bw))
            }
            BQFParam::S(s) => w0_sin / 2.0 * ((a + 1.0 / a) * (1.0 / s - 1.0) + 2.0).sqrt(),
        };
        let (b0, b1, b2, a0, a1, a2);
        match filter_type {
            BQFType::LowPass => {
                b0 = (1.0 - w0_cos) / 2.0;
                b1 = 1.0 - w0_cos;
                b2 = (1.0 - w0_cos) / 2.0;
                a0 = 1.0 + alpha;
                a1 = -2.0 * w0_cos;
                a2 = 1.0 - alpha;
            }
            BQFType::HighPass => {
                b0 = (1.0 + w0_cos) / 2.0;
                b1 = -1.0 - w0_cos;
                b2 = (1.0 + w0_cos) / 2.0;
                a0 = 1.0 + alpha;
                a1 = -2.0 * w0_cos;
                a2 = 1.0 - alpha;
            }
            BQFType::BandPass => {
                b0 = alpha;
                b1 = 0.0;
                b2 = -1.0 * alpha;
                a0 = 1.0 + alpha;
                a1 = -2.0 * w0_cos;
                a2 = 1.0 - alpha;
            }
            BQFType::Notch => {
                b0 = 1.0;
                b1 = -2.0 * w0_cos;
                b2 = 1.0;
                a0 = 1.0 + alpha;
                a1 = -2.0 * w0_cos;
                a2 = 1.0 - alpha;
            }
            BQFType::AudioPeak => {
                b0 = 1.0 - alpha;
                b1 = -2.0 * w0_cos;
                b2 = 1.0 + alpha;
                a0 = 1.0 + alpha;
                a1 = -2.0 * w0_cos;
                a2 = 1.0 - alpha;
            }
            BQFType::PeakingEQ => {
                b0 = 1.0 + alpha * a;
                b1 = -2.0 * w0_cos;
                b2 = 1.0 - alpha * a;
                a0 = 1.0 + alpha / a;
                a1 = -2.0 * w0_cos;
                a2 = 1.0 - alpha / a;
            }
            BQFType::LowShelf => {
                b0 = a * ((a + 1.0) - (a - 1.0) * w0_cos + 2.0 * a.sqrt() * alpha);
                b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * w0_cos);
                b2 = a * ((a + 1.0) - (a - 1.0) * w0_cos - 2.0 * a.sqrt() * alpha);
                a0 = (a + 1.0) + (a - 1.0) * w0_cos + 2.0 * a.sqrt() * alpha;
                a1 = -2.0 * ((a - 1.0) + (a + 1.0) * w0_cos);
                a2 = (a + 1.0) + (a - 1.0) * w0_cos - 2.0 * a.sqrt() * alpha;
            }
            BQFType::HighShelf => {
                b0 = a * ((a + 1.0) - (a - 1.0) * w0_cos + 2.0 * a.sqrt() * alpha);
                b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * w0_cos);
                b2 = a * ((a + 1.0) + (a - 1.0) * w0_cos - 2.0 * a.sqrt() * alpha);
                a0 = (a + 1.0) - (a - 1.0) * w0_cos + 2.0 * a.sqrt() * alpha;
                a1 = 2.0 * ((a - 1.0) - (a + 1.0) * w0_cos);
                a2 = (a + 1.0) - (a - 1.0) * w0_cos - 2.0 * a.sqrt() * alpha;
            }
        }
        BQFCoeff {
            b0,
            b1,
            b2,
            a0,
            a1,
            a2,
            b0_div_a0: b0 / a0,
            b1_div_a0: b1 / a0,
            b2_div_a0: b2 / a0,
            a1_div_a0: a1 / a0,
            a2_div_a0: a2 / a0,
        }
    }

    pub fn dump(&self) -> String {
        format!(
            "b\n{}\n{}\n{}\na\n{}\n{}\n{}\n",
            self.b0, self.b1, self.b2, self.a0, self.a1, self.a2
        )
    }
}

#[derive(Debug)]
pub struct BiquadFilter {
    /// filter type
    filter_type: BQFType,
    /// sampling rate
    rate: f32,
    /// cutoff/center freq
    f0: f32,
    /// gain in dB (PeakingEQ | LowShelf | HighShelf)     
    gain: f32,
    /// Q/Bandwidth/Slope value
    param: BQFParam,
    /// input delay buffer
    buf_x: [f32; 2],
    /// output delay buffer
    buf_y: [f32; 2],
    /// coefficients
    coeff: BQFCoeff,
}

impl BiquadFilter {
    pub fn new(filter_type: BQFType, rate: f32, f0: f32, gain: f32, param: BQFParam) -> Self {
        let coeff = BQFCoeff::new(&filter_type, rate, f0, gain, &param);
        let bqf = BiquadFilter {
            filter_type,
            rate,
            f0,
            gain,
            param,
            buf_x: [0.0; 2],
            buf_y: [0.0; 2],
            coeff,
        };
        log::debug!("BiquadFilter::new {:?}", bqf);
        bqf
    }
    pub fn newb(filter_type: BQFType, rate: f32, f0: f32, gain: f32, param: BQFParam) -> Box<Self> {
        Box::new(Self::new(filter_type, rate, f0, gain, param))
    }
}

impl Filter for BiquadFilter {
    fn apply(&mut self, samples: Vec<f32>) -> Vec<f32> {
        let mut buf: Vec<f32> = Vec::with_capacity(samples.len());
        for x in samples.into_iter() {
            let y = self.coeff.b0_div_a0 * x
                + self.coeff.b1_div_a0 * self.buf_x[0]
                + self.coeff.b2_div_a0 * self.buf_x[1]
                - self.coeff.a1_div_a0 * self.buf_y[0]
                - self.coeff.a2_div_a0 * self.buf_y[1];
            // delay
            self.buf_x[1] = self.buf_x[0];
            self.buf_x[0] = x;
            self.buf_y[1] = self.buf_y[0];
            self.buf_y[0] = y;
            buf.push(y);
        }
        buf
    }
}

pub fn dump_coeffs(mut v: Vec<BiquadFilter>) -> String {
    v.iter_mut()
        .fold(String::new(), |acc, x| format!("{}{}", acc, x.coeff.dump()))
}

pub fn nextpow2(n: f32) -> usize {
    2.0f32.powf(n.log2().ceil()) as usize
}

pub fn generate_impulse(n: usize) -> Vec<f32> {
    let mut buf: Vec<f32> = vec![0.0; n];
    buf[0] = 1.0;
    buf
}

pub fn dump_ir(mut v: Vec<Box<dyn Filter>>, n: usize) -> String {
    let buf = generate_impulse(n);
    let buf = v.iter_mut().fold(buf, |x, f| f.apply(x));
    buf.iter()
        .fold(String::new(), |acc, &x| acc + &x.to_string() + "\n")
        .to_string()
}
