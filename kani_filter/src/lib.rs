use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt::Debug;

/// [-32768, 32767] -> [-1.0, 1.0]
pub fn i16_to_f32(a: &[i16]) -> Vec<f32> {
    a.iter()
        .map(|x| {
            if *x < 0 {
                *x as f32 / 0x8000 as f32
            } else {
                *x as f32 / 0x7fff as f32
            }
        })
        .collect()
}

/// [-1.0, 1.0] -> [-32768, 32767]
pub fn f32_to_i16(a: &[f32]) -> Vec<i16> {
    a.iter()
        .map(|x| {
            if *x < 0.0 {
                (x * 0x8000 as f32) as i16
            } else {
                (x * 0x7fff as f32) as i16
            }
        })
        .collect()
}

/// Separates an interleaved vector.
///
/// # Panics
///
/// Panics if `a.len()` is odd.
pub fn from_interleaved(a: &[f32]) -> (Vec<f32>, Vec<f32>) {
    assert!(a.len() % 2 == 0, "a.len() is odd");
    let mut l: Vec<f32> = Vec::with_capacity(a.len() / 2);
    let mut r: Vec<f32> = Vec::with_capacity(a.len() / 2);
    for (i, s) in a.iter().enumerate() {
        if i % 2 == 0 {
            l.push(*s);
        } else {
            r.push(*s);
        }
    }
    (l, r)
}

/// Interleaves two vectors.
///
/// # Panics
///
/// Panics if `l.len()` != `r.len()`.
pub fn to_interleaved(l: &[f32], r: &[f32]) -> Vec<f32> {
    assert_eq!(
        l.len(),
        r.len(),
        "`l` and `r` must have same length but l={} r={}",
        l.len(),
        r.len()
    );
    let mut a: Vec<f32> = Vec::with_capacity(l.len() * 2);
    for (lv, rv) in l.iter().zip(r) {
        a.push(*lv);
        a.push(*rv);
    }
    a
}

pub type BoxedFilter = Box<dyn Filter + Send>;
pub type VecFilters = Vec<BoxedFilter>;
pub type BoxedFilter2ch = Box<dyn Filter2ch + Send>;
pub type VecFilters2ch = Vec<BoxedFilter2ch>;

/// FilterType includes all filters and used for serialize/deserialize.
/// Each Filter has `_ft` field to hold this value.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
enum FilterType {
    // 1ch
    FTNop,
    FTDelay,
    FTVolume,
    FTBiquad,
    FTConvolver,
    FTReverbBeta,
    // 2ch
    FTPaired,
    FTVocalRemover,
    FTCrossfeedBeta,
}

impl Default for FilterType {
    fn default() -> Self {
        FilterType::FTNop
    }
}

pub trait Filter {
    fn apply(&mut self, xs: &[f32]) -> Vec<f32>;
    fn to_json(&self) -> String;
    fn init(&mut self, fs: f32);
}

impl Debug for dyn Filter + Send {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "filter::Filter(dyn) {}", self.to_json())
    }
}

impl Debug for dyn Filter2ch + Send {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "filter::Filter2ch(dyn) {}", self.to_json())
    }
}

// Note: currently no way to get sampling rate from self
// impl Clone for BoxedFilter {
//     fn clone(&self) -> BoxedFilter {
//         let j = self.to_json();
//         let mut cloned = json_to_filter(&j);
//         cloned.init();
//         cloned
//     }
// }

pub trait Filter2ch {
    fn apply(&mut self, l: &[f32], r: &[f32]) -> (Vec<f32>, Vec<f32>);
    fn to_json(&self) -> String;
    fn init(&mut self, fs: f32);
}

impl Debug for dyn Filter2ch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "filter::Filter2ch(dyn) {}", self.to_json())
    }
}

// Note: currently no way to get sampling rate from self
// impl Clone for BoxedFilter2ch {
//     fn clone(&self) -> BoxedFilter2ch {
//         let j = self.to_json();
//         let mut cloned = json_to_filter2ch(&j);
//         cloned.init();
//         cloned
//     }
// }

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct Delay {
    _ft: FilterType,
    #[serde(skip)]
    sample_rate: f32,
    time_ms: usize,
    #[serde(skip)]
    tapnum: usize,
    #[serde(skip)]
    #[serde(default = "VecDeque::new")]
    buf: VecDeque<f32>,
}

impl Delay {
    const MAX_TAPNUM: usize = 1 << 30 >> 2; // 1GiB f32 array
    /// Initializes Delay with given duration in millisecond and sampling rate.
    ///
    /// # Panics
    ///
    /// Panics if (time_ms*sample_rate/1000) > (2^28) as it needs more than 1GiB mem.
    /// FYI: this means >699000ms delay at 384kHz :D
    pub fn new(time_ms: usize, sample_rate: f32) -> Self {
        let mut f = Self {
            _ft: FilterType::FTDelay,
            time_ms,
            ..Default::default()
        };
        f.init(sample_rate);
        f
    }
    pub fn init(&mut self, fs: f32) {
        self.sample_rate = fs;
        self.tapnum = (self.time_ms as i64 * self.sample_rate as i64 / 1000) as usize;
        assert!(self.tapnum <= Self::MAX_TAPNUM, "too long duration");
        log::trace!("delay tapnum={}", self.tapnum);
        self.buf = VecDeque::from(vec![0.0; self.tapnum]);
        log::debug!(
            "filter::Delay time_ms={} sample_rate={} tapnum={}",
            self.time_ms,
            self.sample_rate,
            self.tapnum
        );
    }
    pub fn newb(time_ms: usize, sample_rate: f32) -> BoxedFilter {
        Box::new(Self::new(time_ms, sample_rate))
    }
    pub fn apply(&mut self, xs: &[f32]) -> Vec<f32> {
        if self.tapnum == 0 {
            return xs.to_vec(); // do nothing
        }
        let mut y = Vec::with_capacity(xs.len());
        for x in xs.iter() {
            y.push(self.buf.pop_front().unwrap()); // already initialized in constructor
            self.buf.push_back(*x);
        }
        y
    }
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
    pub fn from_json(s: &str) -> Self {
        serde_json::from_str(s).unwrap()
    }
}

impl Filter for Delay {
    fn apply(&mut self, xs: &[f32]) -> Vec<f32> {
        self.apply(xs)
    }
    fn to_json(&self) -> String {
        self.to_json()
    }
    fn init(&mut self, fs: f32) {
        self.init(fs);
    }
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct ReverbBeta {
    _ft: FilterType,
    #[serde(skip)]
    sample_rate: f32,
    #[serde(skip)]
    d1: Delay,
    #[serde(skip)]
    d2: Delay,
    #[serde(skip)]
    d3: Delay,
    #[serde(skip)]
    d4: Delay,
    #[serde(skip)]
    d5: Delay,
    #[serde(skip)]
    d6: Delay,
}

impl ReverbBeta {
    pub fn new(sample_rate: f32) -> Self {
        let mut f = Self {
            _ft: FilterType::FTReverbBeta,
            // time_ms,
            ..Default::default()
        };
        f.init(sample_rate);
        f
    }
    pub fn init(&mut self, fs: f32) {
        log::debug!("filter::ReverbBeta");
        self.sample_rate = fs;
        let primes = [113, 229, 349, 463, 601, 733, 863, 1013];
        self.d1 = Delay::new(primes[0], self.sample_rate);
        self.d2 = Delay::new(primes[1], self.sample_rate);
        self.d3 = Delay::new(primes[2], self.sample_rate);
        self.d4 = Delay::new(primes[3], self.sample_rate);
        self.d5 = Delay::new(primes[4], self.sample_rate);
        self.d6 = Delay::new(primes[5], self.sample_rate);
    }
    pub fn newb(sample_rate: f32) -> BoxedFilter {
        Box::new(Self::new(sample_rate))
    }
    pub fn apply(&mut self, xs: &[f32]) -> Vec<f32> {
        let y1 = self.d1.apply(xs);
        let y2 = self.d2.apply(xs);
        let y3 = self.d3.apply(xs);
        let y4 = self.d4.apply(xs);
        let y5 = self.d5.apply(xs);
        let y6 = self.d6.apply(xs);
        let mut y = Vec::with_capacity(xs.len());
        for i in 0..xs.len() {
            // 1*0.3*0.4*0.5*0.6*0.7*0.8
            let val = xs[i]
                + y1[i] * 0.3
                + y2[i] * 0.12
                + y3[i] * 0.06
                + y4[i] * 0.036
                + y5[i] * 0.0252
                + y6[i] * 0.02016;
            y.push(val * 0.64 * 1.12); // 1.12 = +1dB
        }
        y
    }
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
    pub fn from_json(s: &str) -> Self {
        serde_json::from_str(s).unwrap()
    }
}

impl Filter for ReverbBeta {
    fn apply(&mut self, xs: &[f32]) -> Vec<f32> {
        self.apply(xs)
    }
    fn to_json(&self) -> String {
        self.to_json()
    }
    fn init(&mut self, fs: f32) {
        self.init(fs);
    }
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct Volume {
    _ft: FilterType,
    curve: VolumeCurve,
    val: f32,
    #[serde(skip)]
    ratio: f32,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum VolumeCurve {
    Linear,
    Gain,
}

impl Default for VolumeCurve {
    fn default() -> Self {
        VolumeCurve::Linear
    }
}

impl Volume {
    pub fn new(curve: VolumeCurve, val: f32) -> Self {
        let mut f = Self {
            _ft: FilterType::FTVolume,
            curve,
            val,
            ..Default::default()
        };
        f.init();
        f
    }
    pub fn init(&mut self) {
        self.ratio = match self.curve {
            VolumeCurve::Linear => self.val,
            VolumeCurve::Gain => 10.0f32.powf(self.val / 20.0),
        };
        log::debug!(
            "filter::Volume {:?}({})=>{}",
            self.curve,
            self.val,
            self.ratio
        );
    }
    pub fn newb(curve: VolumeCurve, val: f32) -> BoxedFilter {
        Box::new(Self::new(curve, val))
    }
    pub fn apply(&mut self, xs: &[f32]) -> Vec<f32> {
        xs.iter().map(|x| x * self.ratio as f32).collect()
    }
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
    pub fn from_json(s: &str) -> Self {
        serde_json::from_str(s).unwrap()
    }
}

impl Filter for Volume {
    fn apply(&mut self, xs: &[f32]) -> Vec<f32> {
        self.apply(xs)
    }
    fn to_json(&self) -> String {
        self.to_json()
    }
    fn init(&mut self, _: f32) {
        self.init();
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
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

impl Default for BQFType {
    fn default() -> Self {
        BQFType::LowPass
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum BQFParam {
    // Q factor
    Q(f32),
    // Bandwidth (Octave)
    BW(f32),
    // Slope
    S(f32),
}

impl Default for BQFParam {
    fn default() -> Self {
        BQFParam::Q(0.707)
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize, Default)]
pub struct BQFCoeff {
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
                b0 = a * ((a + 1.0) + (a - 1.0) * w0_cos + 2.0 * a.sqrt() * alpha);
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

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct BiquadFilter {
    _ft: FilterType,
    /// BQF type
    filter_type: BQFType,
    /// sampling rate
    #[serde(skip)]
    rate: f32,
    /// cutoff/center freq
    f0: f32,
    /// gain in dB (PeakingEQ | LowShelf | HighShelf)
    gain: f32,
    /// Q/Bandwidth/Slope value
    param: BQFParam,
    /// input delay buffer
    #[serde(skip)]
    buf_x: [f32; 2],
    /// output delay buffer
    #[serde(skip)]
    buf_y: [f32; 2],
    /// coefficients
    #[serde(skip)]
    coeff: BQFCoeff,
}

impl BiquadFilter {
    /// Biquad Filter implementation based on [RBJ Cookbook](https://webaudio.github.io/Audio-EQ-Cookbook/Audio-EQ-Cookbook.txt)
    pub fn new(filter_type: BQFType, rate: f32, f0: f32, gain: f32, param: BQFParam) -> Self {
        let mut f = Self {
            _ft: FilterType::FTBiquad,
            filter_type,
            f0,
            gain,
            param,
            ..Default::default()
        };
        f.init(rate);
        f
    }
    pub fn init(&mut self, rate: f32) {
        self.rate = rate;

        // validation
        if self.f0 >= self.rate / 2.0 {
            self.f0 = (self.rate / 2.0) - 2.0;
        }
        if self.f0 <= 0.0 {
            self.f0 = 2.0;
        }
        self.coeff = BQFCoeff::new(
            &self.filter_type,
            self.rate,
            self.f0,
            self.gain,
            &self.param,
        );
        log::debug!("filter::BiquadFilter {:?}", self);
    }
    pub fn newb(
        filter_type: BQFType,
        rate: f32,
        f0: f32,
        gain: f32,
        param: BQFParam,
    ) -> BoxedFilter {
        Box::new(Self::new(filter_type, rate, f0, gain, param))
    }
    pub fn apply(&mut self, xs: &[f32]) -> Vec<f32> {
        let mut buf: Vec<f32> = Vec::with_capacity(xs.len());
        for x in xs.iter() {
            let y = self.coeff.b0_div_a0 * x
                + self.coeff.b1_div_a0 * self.buf_x[0]
                + self.coeff.b2_div_a0 * self.buf_x[1]
                - self.coeff.a1_div_a0 * self.buf_y[0]
                - self.coeff.a2_div_a0 * self.buf_y[1];
            // delay
            self.buf_x[1] = self.buf_x[0];
            self.buf_x[0] = *x;
            self.buf_y[1] = self.buf_y[0];
            self.buf_y[0] = y;
            buf.push(y);
        }
        buf
    }
    pub fn dump_coeff(&self) -> String {
        self.coeff.dump()
    }
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
    pub fn from_json(s: &str) -> Self {
        serde_json::from_str(s).unwrap()
    }
}

impl Filter for BiquadFilter {
    fn apply(&mut self, xs: &[f32]) -> Vec<f32> {
        self.apply(xs)
    }
    fn to_json(&self) -> String {
        self.to_json()
    }
    fn init(&mut self, fs: f32) {
        self.init(fs);
    }
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct Convolver {
    _ft: FilterType,
    /// impulse response
    ir: Vec<f32>,
    /// inverse(ir)
    #[serde(skip)]
    iir: Vec<f32>,
    /// ring buffer
    #[serde(skip)]
    buf: VecDeque<f32>,
}

impl Convolver {
    /// set small `max_n` in load_ir() to limit impulse response length helps performance
    /// Tested in Core i7-4790:
    ///   n=3072 2ch convolver uses 35% CPU in release build (n>=4096 causes underrun)
    ///   n=96 2ch convolver uses 60% CPU in debug build (n>=128 causes underrun)
    pub fn new(ir: &[f32]) -> Self {
        let mut f = Self {
            _ft: FilterType::FTConvolver,
            ir: ir.to_vec(),
            ..Default::default()
        };
        f.init();
        f
    }
    pub fn init(&mut self) {
        log::debug!("filter::Convolver ir.len()={}", self.ir.len());
        self.iir = self.ir.clone();
        self.iir.reverse();
        self.buf = VecDeque::from(vec![0.0; self.ir.len()]);
    }
    pub fn newb(ir: &[f32]) -> BoxedFilter {
        Box::new(Self::new(ir))
    }
    pub fn apply(&mut self, xs: &[f32]) -> Vec<f32> {
        if self.iir.is_empty() {
            return xs.to_vec(); // do nothing
        }
        let mut vy: Vec<f32> = Vec::with_capacity(xs.len());
        // convolve([1, 3, 7], [1, -10, 100]) -> [1, -7, 77]
        // initial      [0 0 0]1 3 7
        // [1]       <-  0[0 0 1]3 7
        // [1 -7]    <-  0 0[0 1 3]7
        // [1 -7 77] <-  0 0 0[1 3 7]
        for input_x in xs.iter() {
            self.buf.pop_front().unwrap();
            self.buf.push_back(*input_x);
            let mut y = 0.0f32;
            for (x, k) in self.buf.iter().zip(&self.iir) {
                y += x * k;
            }
            vy.push(y);
        }
        vy
    }
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
    pub fn from_json(s: &str) -> Self {
        serde_json::from_str(s).unwrap()
    }
}

impl Filter for Convolver {
    fn apply(&mut self, xs: &[f32]) -> Vec<f32> {
        self.apply(xs)
    }
    fn to_json(&self) -> String {
        self.to_json()
    }
    fn init(&mut self, _: f32) {
        self.init();
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum VocalRemoverType {
    RemoveCenter,
    /// (Remove start, Remove end)
    /// e.g., RemoveCenterBW(220.0, 4400.0)
    RemoveCenterBW(f32, f32),
}

impl Default for VocalRemoverType {
    fn default() -> Self {
        VocalRemoverType::RemoveCenter
    }
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct VocalRemover {
    _ft: FilterType,
    vrtype: VocalRemoverType,
    #[serde(skip)]
    fs: f32,
    #[serde(skip)]
    lv: Volume,
    #[serde(skip)]
    rv: Volume,
    #[serde(skip)]
    ll: BiquadFilter,
    #[serde(skip)]
    llx: BiquadFilter,
    #[serde(skip)]
    lh: BiquadFilter,
    #[serde(skip)]
    lhx: BiquadFilter,
    #[serde(skip)]
    rl: BiquadFilter,
    #[serde(skip)]
    rlx: BiquadFilter,
    #[serde(skip)]
    rh: BiquadFilter,
    #[serde(skip)]
    rhx: BiquadFilter,
}

impl VocalRemover {
    const VOL: f32 = -3.0;

    // const RL: f32 = 0.707;
    // const RH: f32 = 1.0 / Self::RL;
    // const P: BQFParam = BQFParam::Q(0.707);

    const RL: f32 = 0.882;
    const RH: f32 = 1.0 / Self::RL;
    const P: BQFParam = BQFParam::Q(1.414);

    pub fn new(vrtype: VocalRemoverType, fs: f32) -> Self {
        let mut f = Self {
            _ft: FilterType::FTVocalRemover,
            vrtype,
            ..Default::default()
        };
        f.init(fs);
        f
    }
    pub fn init(&mut self, fs: f32) {
        self.fs = fs;

        log::debug!("filter::VocalRemover type={:?}", self.vrtype);
        match self.vrtype {
            VocalRemoverType::RemoveCenterBW(fl, fh) => {
                self.lv = Volume::new(VolumeCurve::Gain, Self::VOL);
                self.rv = Volume::new(VolumeCurve::Gain, Self::VOL);
                self.ll = BiquadFilter::new(BQFType::LowPass, self.fs, fl * Self::RL, 0.0, Self::P);
                self.llx =
                    BiquadFilter::new(BQFType::HighPass, self.fs, fl * Self::RH, 0.0, Self::P);
                self.lh =
                    BiquadFilter::new(BQFType::HighPass, self.fs, fh * Self::RH, 0.0, Self::P);
                self.lhx =
                    BiquadFilter::new(BQFType::LowPass, self.fs, fh * Self::RL, 0.0, Self::P);
                self.rl = BiquadFilter::new(BQFType::LowPass, self.fs, fl * Self::RL, 0.0, Self::P);
                self.rlx =
                    BiquadFilter::new(BQFType::HighPass, self.fs, fl * Self::RH, 0.0, Self::P);
                self.rh =
                    BiquadFilter::new(BQFType::HighPass, self.fs, fh * Self::RH, 0.0, Self::P);
                self.rhx =
                    BiquadFilter::new(BQFType::LowPass, self.fs, fh * Self::RL, 0.0, Self::P);
            }
            _ => {}
        }
    }
    pub fn newb(vrtype: VocalRemoverType, fs: f32) -> BoxedFilter2ch {
        Box::new(Self::new(vrtype, fs))
    }
    pub fn apply(&mut self, l: &[f32], r: &[f32]) -> (Vec<f32>, Vec<f32>) {
        match self.vrtype {
            VocalRemoverType::RemoveCenter => {
                let lr: Vec<f32> = l.iter().zip(r).map(|(l, r)| l - r).collect();
                (lr.clone(), lr)
            }
            VocalRemoverType::RemoveCenterBW(_, _) => {
                // reduce gain to avoid clipping
                let l = self.lv.apply(l);
                let r = self.rv.apply(r);

                let ll = self.ll.apply(&l); // low (L ch)
                let lh = self.lh.apply(&l); // high (L ch)
                let rl = self.rl.apply(&r); // low (R ch)
                let rh = self.rh.apply(&r); // high (R ch)
                let lm = self.lhx.apply(&self.llx.apply(&l)); // mid (L ch)
                let rm = self.rhx.apply(&self.rlx.apply(&r)); // mid (R ch)
                let center: Vec<f32> = lm.iter().zip(&rm).map(|(l, r)| l - r).collect(); // mid (L-R)

                let len = l.len();
                let mut lo = Vec::with_capacity(len); // output (L ch)
                let mut ro = Vec::with_capacity(len); // output (R ch)
                for i in 0..len {
                    let lsum = ll[i] + center[i] + lh[i];
                    let rsum = rl[i] + center[i] + rh[i];
                    #[cfg(debug_assertions)]
                    if lsum.abs() > 1.0 || rsum.abs() > 1.0 {
                        log::debug!("clipping detected l={} r={}", lsum, rsum);
                    }
                    lo.push(lsum);
                    ro.push(rsum);
                }
                (lo, ro)
            }
        }
    }
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
    pub fn from_json(s: &str) -> Self {
        serde_json::from_str(s).unwrap()
    }
}

impl Filter2ch for VocalRemover {
    fn apply(&mut self, l: &[f32], r: &[f32]) -> (Vec<f32>, Vec<f32>) {
        self.apply(l, r)
    }
    fn to_json(&self) -> String {
        self.to_json()
    }
    fn init(&mut self, fs: f32) {
        self.init(fs);
    }
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct CrossfeedBeta {
    _ft: FilterType,
    basefq: f32,
    #[serde(skip)]
    fs: f32,
    #[serde(skip)]
    ld1: Delay,
    #[serde(skip)]
    ld2: Delay,
    #[serde(skip)]
    ld3: Delay,
    #[serde(skip)]
    ld4: Delay,
    #[serde(skip)]
    ld5: Delay,
    #[serde(skip)]
    ld6: Delay,
    #[serde(skip)]
    rd1: Delay,
    #[serde(skip)]
    rd2: Delay,
    #[serde(skip)]
    rd3: Delay,
    #[serde(skip)]
    rd4: Delay,
    #[serde(skip)]
    rd5: Delay,
    #[serde(skip)]
    rd6: Delay,
    #[serde(skip)]
    ll1: BiquadFilter,
    #[serde(skip)]
    ll2: BiquadFilter,
    #[serde(skip)]
    ll3: BiquadFilter,
    #[serde(skip)]
    ll4: BiquadFilter,
    #[serde(skip)]
    ll5: BiquadFilter,
    #[serde(skip)]
    ll6: BiquadFilter,
    #[serde(skip)]
    rl1: BiquadFilter,
    #[serde(skip)]
    rl2: BiquadFilter,
    #[serde(skip)]
    rl3: BiquadFilter,
    #[serde(skip)]
    rl4: BiquadFilter,
    #[serde(skip)]
    rl5: BiquadFilter,
    #[serde(skip)]
    rl6: BiquadFilter,
}

impl CrossfeedBeta {
    pub fn new(basefq: f32, fs: f32) -> Self {
        let mut f = Self {
            _ft: FilterType::FTCrossfeedBeta,
            basefq,
            ..Default::default()
        };
        f.init(fs);
        f
    }
    pub fn init(&mut self, fs: f32) {
        log::debug!("filter::CrossfeedBeta");
        self.fs = fs;
        let primes = [113, 229, 349, 463, 601, 733, 863, 1013];
        self.ld1 = Delay::new(primes[0], self.fs);
        self.ld2 = Delay::new(primes[1], self.fs);
        self.ld3 = Delay::new(primes[2], self.fs);
        self.ld4 = Delay::new(primes[3], self.fs);
        self.ld5 = Delay::new(primes[4], self.fs);
        self.ld6 = Delay::new(primes[5], self.fs);
        self.rd1 = Delay::new(primes[0], self.fs);
        self.rd2 = Delay::new(primes[1], self.fs);
        self.rd3 = Delay::new(primes[2], self.fs);
        self.rd4 = Delay::new(primes[3], self.fs);
        self.rd5 = Delay::new(primes[4], self.fs);
        self.rd6 = Delay::new(primes[5], self.fs);
        // attenuation
        // y=0 => 100%, y=1 => 90%, y=2 => 80%, ...
        let a = |x: f32, y: isize| x - (x * 0.1 * y as f32);
        let b = self.basefq;
        self.ll1 = BiquadFilter::new(BQFType::LowPass, self.fs, a(b, 0), 0.0, BQFParam::Q(0.707));
        self.ll2 = BiquadFilter::new(BQFType::LowPass, self.fs, a(b, 1), 0.0, BQFParam::Q(0.707));
        self.ll3 = BiquadFilter::new(BQFType::LowPass, self.fs, a(b, 2), 0.0, BQFParam::Q(0.707));
        self.ll4 = BiquadFilter::new(BQFType::LowPass, self.fs, a(b, 3), 0.0, BQFParam::Q(0.707));
        self.ll5 = BiquadFilter::new(BQFType::LowPass, self.fs, a(b, 4), 0.0, BQFParam::Q(0.707));
        self.ll6 = BiquadFilter::new(BQFType::LowPass, self.fs, a(b, 5), 0.0, BQFParam::Q(0.707));
        self.rl1 = BiquadFilter::new(BQFType::LowPass, self.fs, a(b, 0), 0.0, BQFParam::Q(0.707));
        self.rl2 = BiquadFilter::new(BQFType::LowPass, self.fs, a(b, 1), 0.0, BQFParam::Q(0.707));
        self.rl3 = BiquadFilter::new(BQFType::LowPass, self.fs, a(b, 2), 0.0, BQFParam::Q(0.707));
        self.rl4 = BiquadFilter::new(BQFType::LowPass, self.fs, a(b, 3), 0.0, BQFParam::Q(0.707));
        self.rl5 = BiquadFilter::new(BQFType::LowPass, self.fs, a(b, 4), 0.0, BQFParam::Q(0.707));
        self.rl6 = BiquadFilter::new(BQFType::LowPass, self.fs, a(b, 5), 0.0, BQFParam::Q(0.707));
    }
    pub fn newb(basefq: f32, fs: f32) -> BoxedFilter2ch {
        Box::new(Self::new(basefq, fs))
    }
    pub fn apply(&mut self, l: &[f32], r: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let l1 = self.ld1.apply(&self.ll1.apply(l));
        let l2 = self.ld2.apply(&self.ll2.apply(l));
        let l3 = self.ld3.apply(&self.ll3.apply(l));
        let l4 = self.ld4.apply(&self.ll4.apply(l));
        let l5 = self.ld5.apply(&self.ll5.apply(l));
        let l6 = self.ld6.apply(&self.ll6.apply(l));

        let r1 = self.rd1.apply(&self.rl1.apply(l));
        let r2 = self.rd2.apply(&self.rl2.apply(l));
        let r3 = self.rd3.apply(&self.rl3.apply(l));
        let r4 = self.rd4.apply(&self.rl4.apply(l));
        let r5 = self.rd5.apply(&self.rl5.apply(l));
        let r6 = self.rd6.apply(&self.rl6.apply(l));

        let mut lo = Vec::with_capacity(l.len());
        let mut ro = Vec::with_capacity(l.len());
        for i in 0..l.len() {
            // 1*0.3*0.4*0.5*0.6*0.7*0.8
            let lval = l[i]
                + l1[i] * 0.3
                + l2[i] * 0.12
                + l3[i] * 0.06
                + l4[i] * 0.036
                + l5[i] * 0.0252
                + l6[i] * 0.02016
                // no r1
                + r2[i] * (0.3 / 2.0)
                + r3[i] * (0.12 / 2.0)
                + r4[i] * (0.06 / 2.0)
                + r5[i] * (0.036 / 2.0)
                + r6[i] * (0.0252 / 2.0);
            let rval = r[i]
                + r1[i] * 0.3
                + r2[i] * 0.12
                + r3[i] * 0.06
                + r4[i] * 0.036
                + r5[i] * 0.0252
                + r6[i] * 0.02016
                // no l1
                + l2[i] * (0.3 / 2.0)
                + l3[i] * (0.12 / 2.0)
                + l4[i] * (0.06 / 2.0)
                + l5[i] * (0.036 / 2.0)
                + l6[i] * (0.0252 / 2.0);
            lo.push(lval * 0.64 * 1.12); // 1.12 = +1dB
            ro.push(rval * 0.64 * 1.12);
        }
        return (lo, ro);
    }
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
    pub fn from_json(s: &str) -> Self {
        serde_json::from_str(s).unwrap()
    }
}

impl Filter2ch for CrossfeedBeta {
    fn apply(&mut self, l: &[f32], r: &[f32]) -> (Vec<f32>, Vec<f32>) {
        self.apply(l, r)
    }
    fn to_json(&self) -> String {
        self.to_json()
    }
    fn init(&mut self, fs: f32) {
        self.init(fs);
    }
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct NopFilter {
    _ft: FilterType,
}

impl NopFilter {
    pub fn new() -> Self {
        let mut f = Self {
            _ft: FilterType::FTNop,
        };
        f.init();
        f
    }
    pub fn init(&mut self) {
        log::debug!("filter::NopFilter");
    }
    pub fn newb() -> BoxedFilter {
        Box::new(Self::new())
    }
    pub fn apply(&mut self, xs: &[f32]) -> Vec<f32> {
        xs.to_vec()
    }
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
    pub fn from_json(s: &str) -> Self {
        serde_json::from_str(s).unwrap()
    }
}

impl Filter for NopFilter {
    fn apply(&mut self, xs: &[f32]) -> Vec<f32> {
        self.apply(xs)
    }
    fn to_json(&self) -> String {
        self.to_json()
    }
    fn init(&mut self, _: f32) {
        self.init();
    }
}

#[derive(Debug)]
pub struct PairedFilter {
    _ft: FilterType,
    l: BoxedFilter,
    r: BoxedFilter,
}

impl PairedFilter {
    pub fn new(l: BoxedFilter, r: BoxedFilter, fs: f32) -> Self {
        let mut f = Self {
            _ft: FilterType::FTPaired,
            l,
            r,
        };
        f.init(fs);
        f
    }
    pub fn init(&mut self, fs: f32) {
        log::debug!("filter::PairedFilter l={:?} r={:?}", self.l, self.r);
        self.l.init(fs);
        self.r.init(fs);
    }
    pub fn newb(l: BoxedFilter, r: BoxedFilter, fs: f32) -> BoxedFilter2ch {
        Box::new(Self::new(l, r, fs))
    }
    pub fn apply(&mut self, l: &[f32], r: &[f32]) -> (Vec<f32>, Vec<f32>) {
        (self.l.apply(l), self.r.apply(r))
    }
    pub fn to_json(&self) -> String {
        String::from(r#"{"_ft":"FTPaired","l":"#)
            + &self.l.to_json()
            + r#","r":"#
            + &self.r.to_json()
            + r#"}"#
    }
}

impl Filter2ch for PairedFilter {
    fn apply(&mut self, l: &[f32], r: &[f32]) -> (Vec<f32>, Vec<f32>) {
        self.apply(l, r)
    }
    fn to_json(&self) -> String {
        self.to_json()
    }
    fn init(&mut self, fs: f32) {
        self.init(fs);
    }
}

pub fn vec_to_json(vf: &VecFilters) -> String {
    if vf.len() == 0 {
        return String::from("[]");
    }
    let s = vf
        .iter()
        .fold(String::from("["), |s, f| s + &f.to_json() + ",");
    let s = &s[0..s.len() - 1];
    s.to_string() + "]"
}

pub fn vec2ch_to_json(vf2: &VecFilters2ch) -> String {
    if vf2.len() == 0 {
        return String::from("[]");
    }
    let s = vf2
        .iter()
        .fold(String::from("["), |s, f| s + &f.to_json() + ",");
    let s = &s[0..s.len() - 1];
    s.to_string() + "]"
}

#[derive(Serialize, Deserialize)]
struct _FilterDeserializationStruct {
    _ft: FilterType,
}

#[derive(Serialize, Deserialize)]
struct _PairedFilterDeserializationStruct {
    _ft: FilterType,
    l: serde_json::Value,
    r: serde_json::Value,
}

pub fn json_to_filter(s: &str, fs: f32) -> Result<BoxedFilter> {
    let x: _FilterDeserializationStruct =
        serde_json::from_str(s).with_context(|| format!("could not deserialize Filter {}", s))?;
    match x._ft {
        FilterType::FTNop => {
            let mut f = Box::<NopFilter>::new(
                serde_json::from_str(s)
                    .with_context(|| format!("could not deserialize NopFilter {}", s))?,
            );
            f.init();
            Ok(f)
        }
        FilterType::FTVolume => {
            let mut f = Box::<Volume>::new(
                serde_json::from_str(s)
                    .with_context(|| format!("could not deserialize Volume {}", s))?,
            );
            f.init();
            Ok(f)
        }
        FilterType::FTDelay => {
            let mut f = Box::<Delay>::new(
                serde_json::from_str(s)
                    .with_context(|| format!("could not deserialize Delay {}", s))?,
            );
            f.init(fs);
            Ok(f)
        }
        FilterType::FTReverbBeta => {
            let mut f = Box::<ReverbBeta>::new(
                serde_json::from_str(s)
                    .with_context(|| format!("could not deserialize ReverbBeta {}", s))?,
            );
            f.init(fs);
            Ok(f)
        }
        FilterType::FTBiquad => {
            let mut f = Box::<BiquadFilter>::new(
                serde_json::from_str(s)
                    .with_context(|| format!("could not deserialize BiquadFilter {}", s))?,
            );
            f.init(fs);
            Ok(f)
        }
        FilterType::FTConvolver => {
            let mut f = Box::<Convolver>::new(
                serde_json::from_str(s)
                    .with_context(|| format!("could not deserialize Convolver {}", s))?,
            );
            f.init();
            Ok(f)
        }
        _ => {
            bail!("could not deserialize unsupported FilterType={:?}", x._ft);
        }
    }
}

pub fn json_to_filter2ch(s: &str, fs: f32) -> Result<BoxedFilter2ch> {
    let x: _FilterDeserializationStruct = serde_json::from_str(s)
        .with_context(|| format!("could not deserialize Filter2ch {}", s))?;
    match x._ft {
        FilterType::FTVocalRemover => {
            let mut f = Box::<VocalRemover>::new(
                serde_json::from_str(s)
                    .with_context(|| format!("could not deserialize VocalRemover {}", s))?,
            );
            f.init(fs);
            Ok(f)
        }
        FilterType::FTCrossfeedBeta => {
            let mut f = Box::<CrossfeedBeta>::new(
                serde_json::from_str(s)
                    .with_context(|| format!("could not deserialize CrossfeedBeta {}", s))?,
            );
            f.init(fs);
            Ok(f)
        }
        FilterType::FTPaired => {
            let tmp: _PairedFilterDeserializationStruct = serde_json::from_str(s)
                .with_context(|| format!("could not deserialize PairedFilter {}", s))?;
            let mut f = Box::<PairedFilter>::new(PairedFilter::new(
                json_to_filter(&format!("{}", tmp.l), fs)?,
                json_to_filter(&format!("{}", tmp.r), fs)?,
                fs,
            ));
            f.l.init(fs);
            f.r.init(fs);
            Ok(f)
        }
        _ => {
            bail!("could not deserialize unsupported FilterType={:?}", x._ft)
        }
    }
}

pub fn json_to_vec(s: &str, fs: f32) -> Result<VecFilters> {
    let mut vf = VecFilters::new();
    let v: Vec<serde_json::Value> =
        serde_json::from_str(s).with_context(|| "could not deserialize VecFilters")?;
    for x in v {
        vf.push(json_to_filter(&format!("{}", x), fs)?);
    }
    Ok(vf)
}

pub fn json_to_vec2ch(s: &str, fs: f32) -> Result<VecFilters2ch> {
    let mut vf = VecFilters2ch::new();
    let v: Vec<serde_json::Value> = serde_json::from_str(s)
        .with_context(|| format!("could not deserialize VecFilters2ch {}", s))?;
    for x in v {
        vf.push(json_to_filter2ch(&format!("{}", x), fs)?);
    }
    Ok(vf)
}

pub fn dump_coeffs(v: &[BiquadFilter]) -> String {
    v.iter()
        .fold(String::new(), |s, x| format!("{}{}", s, x.dump_coeff()))
}

pub fn nextpow2(n: f32) -> usize {
    2.0f32.powf(n.log2().ceil()) as usize
}

pub fn generate_impulse(n: usize) -> Vec<f32> {
    let mut buf: Vec<f32> = vec![0.0; n];
    buf[0] = 1.0;
    buf
}

pub fn generate_inverse(a: &[f32]) -> Vec<f32> {
    a.iter().map(|x| *x * -1.0).collect()
}

pub fn dump_ir(v: &mut Vec<BoxedFilter>, n: usize) -> String {
    let buf = generate_impulse(n);
    let buf = v.iter_mut().fold(buf, |x, f| f.apply(&x));
    buf.iter()
        .fold(String::new(), |s, &x| s + &x.to_string() + "\n")
}

pub fn load_ir(s: &str, max_n: usize) -> Vec<f32> {
    let mut v = Vec::new();
    for (idx, l) in s.lines().enumerate() {
        match l.parse::<f32>() {
            Ok(a) => {
                if v.len() < max_n {
                    v.push(a);
                }
            }
            Err(e) => {
                log::info!("load_ir ignored line={} value={} err={:?}", idx + 1, l, e);
            }
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let t = [0, 1, 2, 3, -1, -2, -3, std::i16::MAX, std::i16::MIN];
        let got = i16_to_f32(&t);
        assert!(got.iter().zip(&want).all(|(a, b)| a == b));
    }

    #[test]
    fn test_f32_to_i16() {
        let want = [
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
        let t = [
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
        let got = f32_to_i16(&t);
        assert!(got.iter().zip(&want).all(|(a, b)| a == b));
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_from_interleaved() {
        let want_l = [0.1, 0.3, 0.5, 0.7];
        let want_r = [0.2, 0.4, 0.6, 0.8];
        let t = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let (got_l, got_r) = from_interleaved(&t);
        assert!(got_l.iter().zip(&want_l).all(|(a, b)| a == b));
        assert!(got_r.iter().zip(&want_r).all(|(a, b)| a == b));
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_to_interleaved() {
        let want = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let t_l = [0.1, 0.3, 0.5, 0.7];
        let t_r = [0.2, 0.4, 0.6, 0.8];
        let got = to_interleaved(&t_l, &t_r);
        assert!(got.iter().zip(&want).all(|(a, b)| a == b));
    }

    #[test]
    #[should_panic]
    fn test_to_interleaved_invalid_vec() {
        let t_l = [0.1, 0.3, 0.5, 0.7];
        let t_r = [0.2, 0.4, 0.6];
        to_interleaved(&t_l, &t_r);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_delay() {
        let want = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
            0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23,
            0.24,
            // 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, // Note: the rest is still in the buffer
        ];
        let t = [
            0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14,
            0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28,
            0.29, 0.30,
        ];
        let got = Delay::new(6, 1000.0).apply(&t);
        assert!(got.iter().zip(&want).all(|(a, b)| a == b));
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_delay_zero() {
        let want = [0.01, 0.02, 0.03, 0.04];
        let t = [0.01, 0.02, 0.03, 0.04];
        let got = Delay::new(0, 1000.0).apply(&t);
        assert!(got.iter().zip(&want).all(|(a, b)| a == b));
    }

    #[test]
    #[should_panic]
    fn test_delay_too_long() {
        Delay::new(Delay::MAX_TAPNUM + 1, 1000.0).apply(&[0.01, 0.02, 0.03, 0.04]);
    }

    #[test]
    #[ignore]
    fn check_delay_mem() {
        // this test needs more than 4GiB RAM (1GiB for x, Delay.buf, y, respectively, and 1GiB for somewhere else ?_?)
        // `cargo test -- --ignored` to run
        let tapsize = 1 << 30 >> 2;
        let x = vec![0.123; tapsize];
        let _y = Delay::new(Delay::MAX_TAPNUM, 1000.0).apply(&x);

        use std::{thread, time};
        let t = time::Duration::from_millis(5000);
        thread::sleep(t);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_volume_linear() {
        let want = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5];
        let t = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let got = Volume::new(VolumeCurve::Linear, 0.5).apply(&t);
        assert!(got.iter().zip(&want).all(|(a, b)| a == b));
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_volume_gain() {
        let t = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let g_n6 = 0.5011872f32;
        let want: Vec<f32> = t.iter().map(|x| x * g_n6).collect();
        let got = Volume::new(VolumeCurve::Gain, -6.0).apply(&t);
        assert!(got.iter().zip(&want).all(|(a, b)| a == b));
    }

    #[test]
    fn test_convolver() {
        let ir = [0.5, -0.3, 0.7, -0.2, 0.4, -0.6, -0.1, -0.01];
        let t1 = [0.01, 0.03, 0.07];
        let t2 = [0.11, 0.13, 0.17];
        let want1 = [0.005, 0.012, 0.033];
        let want2 = [0.053, 0.079, 0.115];
        let mut f = Convolver::new(&ir);
        let got1 = f.apply(&t1);
        let got2 = f.apply(&t2);
        let em = 0.00000001; // enough precision?
        assert!(got1.iter().zip(&want1).all(|(a, b)| (a - b).abs() < em));
        assert!(got2.iter().zip(&want2).all(|(a, b)| (a - b).abs() < em));
    }

    #[test]
    fn test_convolver_empty() {
        let ir = [];
        let t = [0.01, 0.03, 0.07];
        let want = [0.01, 0.03, 0.07];
        let got = Convolver::new(&ir).apply(&t);
        assert!(format!("{:?}", want) == format!("{:?}", got));
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_vocal_remover_remove_center() {
        let t1 = [0.01, 0.02, 0.07];
        let t2 = [0.11, 0.04, 0.17];
        let want1 = [-0.10, -0.02, -0.10];
        let want2 = [-0.10, -0.02, -0.10];
        let mut f = VocalRemover::new(VocalRemoverType::RemoveCenter, 48000.0);
        let (got1, got2) = f.apply(&t1, &t2);
        assert!(got1.iter().zip(&want1).all(|(a, b)| a == b));
        assert!(got2.iter().zip(&want2).all(|(a, b)| a == b));
    }

    #[test]
    fn test_nextpow2() {
        assert!(nextpow2(1023.1f32) == 1024);
        assert!(nextpow2(1024.1f32) == 2048);
    }

    #[test]
    fn test_generate_impulse() {
        assert!(
            format!("{:?}", generate_impulse(6)) == format!("{:?}", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        );
    }

    #[test]
    fn test_generate_inverse() {
        let want = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6];
        let got = generate_inverse(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        assert!(format!("{:?}", got) == format!("{:?}", want));
    }

    #[test]
    fn test_load_ir() {
        let s = "1\n0.15\n0.39\n-0.34916812\n\naaaaaa\n1.0000000E-1\n\n\n\n";
        let want = [1.0, 0.15, 0.39, -0.34916812, 0.10];
        let got = load_ir(s, 4096);
        assert!(format!("{:?}", got) == format!("{:?}", want));

        let s = "1\n0.15\n0.39\n-0.34916812\n\naaaaaa\n1.0000000E-1\n\n\n\n";
        let want = [1.0, 0.15];
        let got = load_ir(s, 2);
        assert!(format!("{:?}", got) == format!("{:?}", want));

        let s = "1";
        let want = [1.0];
        let got = load_ir(s, 4096);
        assert!(format!("{:?}", got) == format!("{:?}", want));

        let s = "";
        let want: [f32; 0] = [];
        let got = load_ir(s, 4096);
        assert!(format!("{:?}", got) == format!("{:?}", want));
    }

    #[test]
    fn test_delay_to_json() {
        let fs = 48000.0;
        let t = Delay::new(200, fs);
        let want = r#"{"_ft":"FTDelay","time_ms":200}"#;
        let got = t.to_json();
        assert_eq!(want, got);
        // deserialize
        let got2 = json_to_filter(want, fs).unwrap().to_json();
        assert_eq!(got, got2);
    }

    #[test]
    fn test_volume_to_json() {
        let fs = 48000.0;
        let t = Volume::new(VolumeCurve::Gain, -6.0);
        let want = r#"{"_ft":"FTVolume","curve":"Gain","val":-6.0}"#;
        let got = t.to_json();
        assert_eq!(want, got);
        // deserialize
        let got2 = json_to_filter(want, fs).unwrap().to_json();
        assert_eq!(got, got2);
    }

    #[test]
    fn test_biquadfilter_to_json() {
        let fs = 48000.0;
        let t = BiquadFilter::new(BQFType::PeakingEQ, fs, 1234.0, -3.5, BQFParam::Q(0.707));
        let want = r#"{"_ft":"FTBiquad","filter_type":"PeakingEQ","f0":1234.0,"gain":-3.5,"param":{"Q":0.707}}"#;
        let got = t.to_json();
        assert_eq!(want, got);
        // deserialize
        let got2 = json_to_filter(want, fs).unwrap().to_json();
        assert_eq!(got, got2);
    }

    #[test]
    fn test_convolver_to_json() {
        let fs = 48000.0;
        let t = Convolver::new(&[0.01, 0.02, 0.03, 0.04]);
        let want = r#"{"_ft":"FTConvolver","ir":[0.01,0.02,0.03,0.04]}"#;
        let got = t.to_json();
        assert_eq!(want, got);
        // deserialize
        let got2 = json_to_filter(want, fs).unwrap().to_json();
        assert_eq!(got, got2);
    }

    #[test]
    fn test_nop_to_json() {
        let fs = 48000.0;
        let t = NopFilter::new();
        let want = r#"{"_ft":"FTNop"}"#;
        let got = t.to_json();
        assert_eq!(want, got);
        // deserialize
        let got2 = json_to_filter(want, fs).unwrap().to_json();
        assert_eq!(got, got2);
    }

    #[test]
    fn test_vocalremover_to_json() {
        let fs = 48000.0;
        let t = VocalRemover::new(VocalRemoverType::RemoveCenterBW(200.0, 4000.0), fs);
        let want = r#"{"_ft":"FTVocalRemover","vrtype":{"RemoveCenterBW":[200.0,4000.0]}}"#;
        let got = t.to_json();
        assert_eq!(want, got);
        // deserialize
        let got2 = json_to_filter2ch(want, fs).unwrap().to_json();
        assert_eq!(got, got2);
    }

    #[test]
    fn test_pairedfilter_to_json() {
        let fs = 48000.0;
        let t = PairedFilter::new(
            Convolver::newb(&[0.01, 0.02, 0.03, 0.04]),
            NopFilter::newb(),
            fs,
        );
        let want = r#"{"_ft":"FTPaired","l":{"_ft":"FTConvolver","ir":[0.01,0.02,0.03,0.04]},"r":{"_ft":"FTNop"}}"#;
        let got = t.to_json();
        assert_eq!(want, got);
        // deserialize
        let got2 = json_to_filter2ch(want, fs).unwrap().to_json();
        assert_eq!(got, got2);
    }

    #[test]
    fn test_vec_to_json() {
        let fs = 48000.0;
        let t = r#"[{"_ft":"FTVolume","curve":"Gain","val":-6.0},{"_ft":"FTDelay","time_ms":200}]"#;
        let got = vec_to_json(&json_to_vec(t, fs).unwrap());
        assert_eq!(t, got);
    }

    #[test]
    fn test_vec2ch_to_json() {
        let fs = 48000.0;
        let t = r#"[{"_ft":"FTPaired","l":{"_ft":"FTVolume","curve":"Gain","val":-6.0},"r":{"_ft":"FTVolume","curve":"Gain","val":-6.0}},{"_ft":"FTVocalRemover","vrtype":{"RemoveCenterBW":[240.0,6600.0]}}]"#;
        let got = vec2ch_to_json(&json_to_vec2ch(t, fs).unwrap());
        assert_eq!(t, got);
    }
}
