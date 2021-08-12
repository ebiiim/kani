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
    // 2ch
    FTPaired,
    FTVocalRemover,
}

pub trait Filter {
    fn apply(&mut self, xs: &[f32]) -> Vec<f32>;
    fn to_json(&self) -> String;
}

impl Debug for dyn Filter {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "filter::Filter(dyn) {}", self.to_json())
    }
}

pub trait Filter2ch {
    fn apply(&mut self, l: &[f32], r: &[f32]) -> (Vec<f32>, Vec<f32>);
    fn to_json(&self) -> String;
}

impl Debug for dyn Filter2ch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "filter::Filter2ch(dyn) {}", self.to_json())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Delay {
    _ft: FilterType,
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
    /// Panics if it needs more than 1GiB mem.
    /// See also: Delay::with_tapnum
    pub fn new(time_ms: usize, sample_rate: usize) -> Self {
        log::debug!(
            "filter::Delay time_ms={} sample_rate={}",
            time_ms,
            sample_rate
        );
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
        assert!(tapnum <= Self::MAX_TAPNUM, "too long duration");
        log::debug!("delay tapnum={}", tapnum);
        let buf: Vec<f32> = vec![0.0; tapnum];
        let buf = VecDeque::from(buf);
        Self {
            _ft: FilterType::FTDelay,
            tapnum,
            buf,
        }
    }
    pub fn newb(time_ms: usize, sample_rate: usize) -> Box<dyn Filter> {
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
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Volume {
    _ft: FilterType,
    curve: VolumeCurve,
    val: f32,
    ratio: f32,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
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
        log::debug!("filter::Volume {:?}({})=>{}", curve, val, ratio);
        Self {
            _ft: FilterType::FTVolume,
            curve,
            val,
            ratio,
        }
    }
    pub fn newb(curve: VolumeCurve, val: f32) -> Box<dyn Filter> {
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

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum BQFParam {
    // Q factor
    Q(f32),
    // Bandwidth (Octave)
    BW(f32),
    // Slope
    S(f32),
}

#[derive(Debug, Serialize, Deserialize)]
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

fn _a2f32() -> [f32; 2] {
    [0.0, 0.0]
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BiquadFilter {
    _ft: FilterType,
    /// BQF type
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
    #[serde(skip)]
    #[serde(default = "_a2f32")]
    buf_x: [f32; 2],
    /// output delay buffer
    #[serde(skip)]
    #[serde(default = "_a2f32")]
    buf_y: [f32; 2],
    /// coefficients
    coeff: BQFCoeff,
}

impl BiquadFilter {
    /// Biquad Filter implementation based on [RBJ Cookbook](https://webaudio.github.io/Audio-EQ-Cookbook/Audio-EQ-Cookbook.txt)
    pub fn new(filter_type: BQFType, rate: f32, f0: f32, gain: f32, param: BQFParam) -> Self {
        // validation
        let mut f0 = f0;
        if f0 >= rate / 2.0 {
            f0 = (rate / 2.0) - 2.0;
        }
        if f0 <= 0.0 {
            f0 = 2.0;
        }

        let coeff = BQFCoeff::new(&filter_type, rate, f0, gain, &param);
        let bqf = BiquadFilter {
            _ft: FilterType::FTBiquad,
            filter_type,
            rate,
            f0,
            gain,
            param,
            buf_x: [0.0; 2],
            buf_y: [0.0; 2],
            coeff,
        };
        log::debug!("filter::BiquadFilter {:?}", bqf);
        bqf
    }
    pub fn newb(
        filter_type: BQFType,
        rate: f32,
        f0: f32,
        gain: f32,
        param: BQFParam,
    ) -> Box<dyn Filter> {
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
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Convolver {
    _ft: FilterType,
    /// impulse response
    ir: Vec<f32>,
    /// ring buffer
    #[serde(skip)]
    #[serde(default = "VecDeque::new")]
    buf: VecDeque<f32>,
}

impl Convolver {
    /// set small `max_n` in load_ir() to limit impulse response length helps performance
    /// Tested in Core i7-4790:
    ///   n=3072 2ch convolver uses 35% CPU in release build (n>=4096 causes underrun)
    ///   n=96 2ch convolver uses 60% CPU in debug build (n>=128 causes underrun)
    pub fn new(ir: &[f32]) -> Self {
        log::debug!("filter::Convolver ir.len()={}", ir.len());
        let buf = vec![0.0; ir.len()];
        let mut ir = ir.to_vec();
        ir.reverse();
        Self {
            _ft: FilterType::FTConvolver,
            ir,
            buf: VecDeque::from(buf),
        }
    }
    pub fn newb(ir: &[f32]) -> Box<dyn Filter> {
        Box::new(Self::new(ir))
    }
    pub fn apply(&mut self, xs: &[f32]) -> Vec<f32> {
        if self.ir.is_empty() {
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
            for (x, k) in self.buf.iter().zip(&self.ir) {
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
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum VocalRemoverType {
    RemoveCenter,
    /// (Sampling rate, Remove start, Remove end)
    /// e.g., RemoveCenterBW(48000.0, 220.0, 4400.0)
    RemoveCenterBW(f32, f32, f32),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VocalRemover {
    _ft: FilterType,
    vrtype: VocalRemoverType,
    lv: Volume,
    rv: Volume,
    ll: BiquadFilter,
    llx: BiquadFilter,
    lh: BiquadFilter,
    lhx: BiquadFilter,
    rl: BiquadFilter,
    rlx: BiquadFilter,
    rh: BiquadFilter,
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

    pub fn new(vrtype: VocalRemoverType) -> Self {
        log::debug!("filter::VocalRemover type={:?}", vrtype);
        match vrtype {
            VocalRemoverType::RemoveCenterBW(fs, fl, fh) => {
                let lv = Volume::new(VolumeCurve::Gain, Self::VOL);
                let rv = Volume::new(VolumeCurve::Gain, Self::VOL);
                let ll = BiquadFilter::new(BQFType::LowPass, fs, fl * Self::RL, 0.0, Self::P);
                let llx = BiquadFilter::new(BQFType::HighPass, fs, fl * Self::RH, 0.0, Self::P);
                let lh = BiquadFilter::new(BQFType::HighPass, fs, fh * Self::RH, 0.0, Self::P);
                let lhx = BiquadFilter::new(BQFType::LowPass, fs, fh * Self::RL, 0.0, Self::P);
                let rl = BiquadFilter::new(BQFType::LowPass, fs, fl * Self::RL, 0.0, Self::P);
                let rlx = BiquadFilter::new(BQFType::HighPass, fs, fl * Self::RH, 0.0, Self::P);
                let rh = BiquadFilter::new(BQFType::HighPass, fs, fh * Self::RH, 0.0, Self::P);
                let rhx = BiquadFilter::new(BQFType::LowPass, fs, fh * Self::RL, 0.0, Self::P);
                Self {
                    _ft: FilterType::FTVocalRemover,
                    vrtype,
                    lv,
                    rv,
                    ll,
                    llx,
                    lh,
                    lhx,
                    rl,
                    rlx,
                    rh,
                    rhx,
                }
            }
            _ => {
                let lv = Volume::new(VolumeCurve::Gain, 0.0);
                let rv = Volume::new(VolumeCurve::Gain, 0.0);
                let ll = BiquadFilter::new(BQFType::LowPass, 0.0, 0.0, 0.0, BQFParam::Q(0.0));
                let llx = BiquadFilter::new(BQFType::LowPass, 0.0, 0.0, 0.0, BQFParam::Q(0.0));
                let lh = BiquadFilter::new(BQFType::LowPass, 0.0, 0.0, 0.0, BQFParam::Q(0.0));
                let lhx = BiquadFilter::new(BQFType::LowPass, 0.0, 0.0, 0.0, BQFParam::Q(0.0));
                let rl = BiquadFilter::new(BQFType::LowPass, 0.0, 0.0, 0.0, BQFParam::Q(0.0));
                let rlx = BiquadFilter::new(BQFType::LowPass, 0.0, 0.0, 0.0, BQFParam::Q(0.0));
                let rh = BiquadFilter::new(BQFType::LowPass, 0.0, 0.0, 0.0, BQFParam::Q(0.0));
                let rhx = BiquadFilter::new(BQFType::LowPass, 0.0, 0.0, 0.0, BQFParam::Q(0.0));
                Self {
                    _ft: FilterType::FTVocalRemover,
                    vrtype,
                    lv,
                    rv,
                    ll,
                    llx,
                    lh,
                    lhx,
                    rl,
                    rlx,
                    rh,
                    rhx,
                }
            }
        }
    }
    pub fn newb(vrtype: VocalRemoverType) -> Box<dyn Filter2ch> {
        Box::new(Self::new(vrtype))
    }
    pub fn apply(&mut self, l: &[f32], r: &[f32]) -> (Vec<f32>, Vec<f32>) {
        match self.vrtype {
            VocalRemoverType::RemoveCenter => {
                let lr: Vec<f32> = l.iter().zip(r).map(|(l, r)| l - r).collect();
                (lr.clone(), lr)
            }
            VocalRemoverType::RemoveCenterBW(_, _, _) => {
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
                        log::warn!("clipping detected l={} r={}", lsum, rsum);
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
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NopFilter {
    _ft: FilterType,
}

impl NopFilter {
    pub fn new() -> Self {
        log::debug!("filter::NopFilter");
        Self {
            _ft: FilterType::FTNop,
        }
    }
    pub fn newb() -> Box<dyn Filter> {
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
}

#[derive(Debug)]
pub struct PairedFilter {
    _ft: FilterType,
    l: Box<dyn Filter>,
    r: Box<dyn Filter>,
}

impl PairedFilter {
    pub fn new(l: Box<dyn Filter>, r: Box<dyn Filter>) -> Self {
        log::debug!("filter::PairedFilter l={:?} r={:?}", l, r);
        Self {
            _ft: FilterType::FTPaired,
            l,
            r,
        }
    }
    pub fn newb(l: Box<dyn Filter>, r: Box<dyn Filter>) -> Box<dyn Filter2ch> {
        Box::new(Self::new(l, r))
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
}

type VecFilters = Vec<Box<dyn Filter>>;

type VecFilters2ch = Vec<Box<dyn Filter2ch>>;

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

pub fn json_to_filter(s: &str) -> Box<dyn Filter> {
    let x: _FilterDeserializationStruct = match serde_json::from_str(s) {
        Ok(v) => v,
        Err(e) => {
            log::error!("json_to_filter returned NopFilter because err={}", e);
            return NopFilter::newb();
        }
    };
    match x._ft {
        FilterType::FTNop => Box::<NopFilter>::new(serde_json::from_str(s).unwrap()),
        FilterType::FTVolume => Box::<Volume>::new(serde_json::from_str(s).unwrap()),
        FilterType::FTDelay => Box::<Delay>::new(serde_json::from_str(s).unwrap()),
        FilterType::FTBiquad => Box::<BiquadFilter>::new(serde_json::from_str(s).unwrap()),
        FilterType::FTConvolver => Box::<Convolver>::new(serde_json::from_str(s).unwrap()),
        _ => {
            log::error!(
                "json_to_filter found unsupported FilterType={:?} and returned NopFilter",
                x._ft
            );
            NopFilter::newb()
        }
    }
}

pub fn json_to_filter2ch(s: &str) -> Box<dyn Filter2ch> {
    let x: _FilterDeserializationStruct = match serde_json::from_str(s) {
        Ok(v) => v,
        Err(e) => {
            log::error!(
                "json_to_filter2 returned PairedFilter(NopFilter,NopFilter) because err={}",
                e
            );
            return PairedFilter::newb(NopFilter::newb(), NopFilter::newb());
        }
    };
    match x._ft {
        FilterType::FTVocalRemover => Box::<VocalRemover>::new(serde_json::from_str(s).unwrap()),
        FilterType::FTPaired => {
            let tmp: _PairedFilterDeserializationStruct = serde_json::from_str(s).unwrap();
            PairedFilter::newb(
                json_to_filter(&format!("{}", tmp.l)),
                json_to_filter(&format!("{}", tmp.r)),
            )
        }
        _ => {
            log::error!(
                "json_to_filter2 found unsupported FilterType={:?} and returned PairedFilter(NopFilter,NopFilter)",
                x._ft
            );
            PairedFilter::newb(NopFilter::newb(), NopFilter::newb())
        }
    }
}

pub fn json_to_vec(s: &str) -> VecFilters {
    let mut vf = VecFilters::new();
    let v: Vec<serde_json::Value> = serde_json::from_str(s).unwrap_or(Vec::new());
    for x in v {
        vf.push(json_to_filter(&format!("{}", x)));
    }
    vf
}

pub fn json_to_vec2ch(s: &str) -> VecFilters2ch {
    let mut vf = VecFilters2ch::new();
    let v: Vec<serde_json::Value> = serde_json::from_str(s).unwrap_or(Vec::new());
    for x in v {
        vf.push(json_to_filter2ch(&format!("{}", x)));
    }
    vf
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

pub fn dump_ir(v: &mut Vec<Box<dyn Filter>>, n: usize) -> String {
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
        let got = Delay::with_tapnum(6).apply(&t);
        assert!(got.iter().zip(&want).all(|(a, b)| a == b));
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_delay_zero() {
        let want = [0.01, 0.02, 0.03, 0.04];
        let t = [0.01, 0.02, 0.03, 0.04];
        let got = Delay::with_tapnum(0).apply(&t);
        assert!(got.iter().zip(&want).all(|(a, b)| a == b));
    }

    #[test]
    #[should_panic]
    fn test_delay_too_long() {
        Delay::with_tapnum(Delay::MAX_TAPNUM + 1).apply(&[0.01, 0.02, 0.03, 0.04]);
    }

    #[test]
    #[ignore]
    fn check_delay_mem() {
        // this test needs more than 4GiB RAM (1GiB for x, Delay.buf, y, respectively, and 1GiB for somewhere else ?_?)
        // `cargo test -- --ignored` to run
        let tapsize = 1 << 30 >> 2;
        let x = vec![0.123; tapsize];
        let _y = Delay::with_tapnum(tapsize).apply(&x);

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
        let mut f = VocalRemover::new(VocalRemoverType::RemoveCenter);
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
        let t = Delay::new(200, 48000);
        let want = r#"{"_ft":"FTDelay","tapnum":9600}"#;
        let got = t.to_json();
        assert_eq!(want, got);
        // deserialize
        let got2 = json_to_filter(want).to_json();
        assert_eq!(got, got2);
    }

    #[test]
    fn test_volume_to_json() {
        let t = Volume::new(VolumeCurve::Gain, -6.0);
        let want = r#"{"_ft":"FTVolume","curve":"Gain","val":-6.0,"ratio":0.5011872}"#;
        let got = t.to_json();
        assert_eq!(want, got);
        // deserialize
        let got2 = json_to_filter(want).to_json();
        assert_eq!(got, got2);
    }

    #[test]
    fn test_biquadfilter_to_json() {
        let t = BiquadFilter::new(
            BQFType::PeakingEQ,
            48000.0,
            1234.0,
            -3.5,
            BQFParam::Q(0.707),
        );
        let want = r#"{"_ft":"FTBiquad","filter_type":"PeakingEQ","rate":48000.0,"f0":1234.0,"gain":-3.5,"param":{"Q":0.707},"coeff":{"b0":1.0929853,"b1":-1.9739647,"b2":0.9070147,"a0":1.1391279,"a1":-1.9739647,"a2":0.86087215,"b0_div_a0":0.95949304,"b1_div_a0":-1.7328737,"b2_div_a0":0.7962361,"a1_div_a0":-1.7328737,"a2_div_a0":0.7557292}}"#;
        let got = t.to_json();
        assert_eq!(want, got);
        // deserialize
        let got2 = json_to_filter(want).to_json();
        assert_eq!(got, got2);
    }

    #[test]
    fn test_convolver_to_json() {
        let t = Convolver::new(&[0.01, 0.02, 0.03, 0.04]);
        let want = r#"{"_ft":"FTConvolver","ir":[0.04,0.03,0.02,0.01]}"#;
        let got = t.to_json();
        assert_eq!(want, got);
        // deserialize
        let got2 = json_to_filter(want).to_json();
        assert_eq!(got, got2);
    }

    #[test]
    fn test_nop_to_json() {
        let t = NopFilter::new();
        let want = r#"{"_ft":"FTNop"}"#;
        let got = t.to_json();
        assert_eq!(want, got);
        // deserialize
        let got2 = json_to_filter(want).to_json();
        assert_eq!(got, got2);
    }

    #[test]
    fn test_vocalremover_to_json() {
        let t = VocalRemover::new(VocalRemoverType::RemoveCenterBW(48000.0, 200.0, 4000.0));
        let want = r#"{"_ft":"FTVocalRemover","vrtype":{"RemoveCenterBW":[48000.0,200.0,4000.0]},"lv":{"_ft":"FTVolume","curve":"Gain","val":-3.0,"ratio":0.70794576},"rv":{"_ft":"FTVolume","curve":"Gain","val":-3.0,"ratio":0.70794576},"ll":{"_ft":"FTBiquad","filter_type":"LowPass","rate":48000.0,"f0":176.40001,"gain":0.0,"param":{"Q":1.414},"coeff":{"b0":0.00013327599,"b1":0.00026655197,"b2":0.00013327599,"a0":1.0081643,"a1":-1.9994669,"a2":0.9918357,"b0_div_a0":0.0001321967,"b1_div_a0":0.0002643934,"b2_div_a0":0.0001321967,"a1_div_a0":-1.9832748,"a2_div_a0":0.98380363}},"llx":{"_ft":"FTBiquad","filter_type":"HighPass","rate":48000.0,"f0":226.75735,"gain":0.0,"param":{"Q":1.414},"coeff":{"b0":0.99977976,"b1":-1.9995595,"b2":0.99977976,"a0":1.0104944,"a1":-1.999119,"a2":0.98950565,"b0_div_a0":0.9893967,"b1_div_a0":-1.9787934,"b2_div_a0":0.9893967,"a1_div_a0":-1.9783574,"a2_div_a0":0.9792293}},"lh":{"_ft":"FTBiquad","filter_type":"HighPass","rate":48000.0,"f0":4535.147,"gain":0.0,"param":{"Q":1.414},"coeff":{"b0":0.9144524,"b1":-1.8289047,"b2":0.9144524,"a0":1.197804,"a1":-1.6578095,"a2":0.802196,"b0_div_a0":0.7634408,"b1_div_a0":-1.5268816,"b2_div_a0":0.7634408,"a1_div_a0":-1.3840407,"a2_div_a0":0.6697223}},"lhx":{"_ft":"FTBiquad","filter_type":"LowPass","rate":48000.0,"f0":3528.0,"gain":0.0,"param":{"Q":1.414},"coeff":{"b0":0.052377164,"b1":0.10475433,"b2":0.052377164,"a0":1.1575576,"a1":-1.7904913,"a2":0.8424424,"b0_div_a0":0.045247998,"b1_div_a0":0.090495996,"b2_div_a0":0.045247998,"a1_div_a0":-1.5467838,"a2_div_a0":0.7277758}},"rl":{"_ft":"FTBiquad","filter_type":"LowPass","rate":48000.0,"f0":176.40001,"gain":0.0,"param":{"Q":1.414},"coeff":{"b0":0.00013327599,"b1":0.00026655197,"b2":0.00013327599,"a0":1.0081643,"a1":-1.9994669,"a2":0.9918357,"b0_div_a0":0.0001321967,"b1_div_a0":0.0002643934,"b2_div_a0":0.0001321967,"a1_div_a0":-1.9832748,"a2_div_a0":0.98380363}},"rlx":{"_ft":"FTBiquad","filter_type":"HighPass","rate":48000.0,"f0":226.75735,"gain":0.0,"param":{"Q":1.414},"coeff":{"b0":0.99977976,"b1":-1.9995595,"b2":0.99977976,"a0":1.0104944,"a1":-1.999119,"a2":0.98950565,"b0_div_a0":0.9893967,"b1_div_a0":-1.9787934,"b2_div_a0":0.9893967,"a1_div_a0":-1.9783574,"a2_div_a0":0.9792293}},"rh":{"_ft":"FTBiquad","filter_type":"HighPass","rate":48000.0,"f0":4535.147,"gain":0.0,"param":{"Q":1.414},"coeff":{"b0":0.9144524,"b1":-1.8289047,"b2":0.9144524,"a0":1.197804,"a1":-1.6578095,"a2":0.802196,"b0_div_a0":0.7634408,"b1_div_a0":-1.5268816,"b2_div_a0":0.7634408,"a1_div_a0":-1.3840407,"a2_div_a0":0.6697223}},"rhx":{"_ft":"FTBiquad","filter_type":"LowPass","rate":48000.0,"f0":3528.0,"gain":0.0,"param":{"Q":1.414},"coeff":{"b0":0.052377164,"b1":0.10475433,"b2":0.052377164,"a0":1.1575576,"a1":-1.7904913,"a2":0.8424424,"b0_div_a0":0.045247998,"b1_div_a0":0.090495996,"b2_div_a0":0.045247998,"a1_div_a0":-1.5467838,"a2_div_a0":0.7277758}}}"#;
        let got = t.to_json();
        assert_eq!(want, got);
        // deserialize
        let got2 = json_to_filter2ch(want).to_json();
        assert_eq!(got, got2);
    }

    #[test]
    fn test_pairedfilter_to_json() {
        let t = PairedFilter::new(
            Convolver::newb(&[0.01, 0.02, 0.03, 0.04]),
            NopFilter::newb(),
        );
        let want = r#"{"_ft":"FTPaired","l":{"_ft":"FTConvolver","ir":[0.04,0.03,0.02,0.01]},"r":{"_ft":"FTNop"}}"#;
        let got = t.to_json();
        assert_eq!(want, got);
        // deserialize
        let got2 = json_to_filter2ch(want).to_json();
        assert_eq!(got, got2);
    }

    #[test]
    fn test_vec_to_json() {
        let t = r#"[{"_ft":"FTVolume","curve":"Gain","val":-6.0,"ratio":0.5011872},{"_ft":"FTDelay","tapnum":8820}]"#;
        let got = vec_to_json(&json_to_vec(t));
        assert_eq!(t, got);
    }

    #[test]
    fn test_vec2ch_to_json() {
        let t = r#"[{"_ft":"FTPaired","l":{"_ft":"FTVolume","curve":"Gain","val":-6.0,"ratio":0.5011872},"r":{"_ft":"FTVolume","curve":"Gain","val":-6.0,"ratio":0.5011872}},{"_ft":"FTVocalRemover","vrtype":{"RemoveCenterBW":[44100.0,240.0,6600.0]},"lv":{"_ft":"FTVolume","curve":"Gain","val":-3.0,"ratio":0.70794576},"rv":{"_ft":"FTVolume","curve":"Gain","val":-3.0,"ratio":0.70794576},"ll":{"_ft":"FTBiquad","filter_type":"LowPass","rate":44100.0,"f0":211.68001,"gain":0.0,"param":{"Q":1.414},"coeff":{"b0":0.00022739172,"b1":0.00045478344,"b2":0.00022739172,"a0":1.0106629,"a1":-1.9990904,"a2":0.9893371,"b0_div_a0":0.00022499265,"b1_div_a0":0.0004499853,"b2_div_a0":0.00022499265,"a1_div_a0":-1.9779992,"a2_div_a0":0.9788992}},"llx":{"_ft":"FTBiquad","filter_type":"HighPass","rate":44100.0,"f0":272.10883,"gain":0.0,"param":{"Q":1.414},"coeff":{"b0":0.99962425,"b1":-1.9992485,"b2":0.99962425,"a0":1.0137055,"a1":-1.9984971,"a2":0.98629445,"b0_div_a0":0.98610914,"b1_div_a0":-1.9722183,"b2_div_a0":0.98610914,"a1_div_a0":-1.971477,"a2_div_a0":0.9729596}},"lh":{"_ft":"FTBiquad","filter_type":"HighPass","rate":44100.0,"f0":7482.9927,"gain":0.0,"param":{"Q":1.414},"coeff":{"b0":0.7417508,"b1":-1.4835016,"b2":0.7417508,"a0":1.3095274,"a1":-0.96700317,"a2":0.6904726,"b0_div_a0":0.56642634,"b1_div_a0":-1.1328527,"b2_div_a0":0.56642634,"a1_div_a0":-0.73843676,"a2_div_a0":0.5272685}},"lhx":{"_ft":"FTBiquad","filter_type":"LowPass","rate":44100.0,"f0":5821.2,"gain":0.0,"param":{"Q":1.414},"coeff":{"b0":0.16233361,"b1":0.32466722,"b2":0.16233361,"a0":1.2607896,"a1":-1.3506656,"a2":0.73921037,"b0_div_a0":0.12875551,"b1_div_a0":0.25751102,"b2_div_a0":0.12875551,"a1_div_a0":-1.0712855,"a2_div_a0":0.58630747}},"rl":{"_ft":"FTBiquad","filter_type":"LowPass","rate":44100.0,"f0":211.68001,"gain":0.0,"param":{"Q":1.414},"coeff":{"b0":0.00022739172,"b1":0.00045478344,"b2":0.00022739172,"a0":1.0106629,"a1":-1.9990904,"a2":0.9893371,"b0_div_a0":0.00022499265,"b1_div_a0":0.0004499853,"b2_div_a0":0.00022499265,"a1_div_a0":-1.9779992,"a2_div_a0":0.9788992}},"rlx":{"_ft":"FTBiquad","filter_type":"HighPass","rate":44100.0,"f0":272.10883,"gain":0.0,"param":{"Q":1.414},"coeff":{"b0":0.99962425,"b1":-1.9992485,"b2":0.99962425,"a0":1.0137055,"a1":-1.9984971,"a2":0.98629445,"b0_div_a0":0.98610914,"b1_div_a0":-1.9722183,"b2_div_a0":0.98610914,"a1_div_a0":-1.971477,"a2_div_a0":0.9729596}},"rh":{"_ft":"FTBiquad","filter_type":"HighPass","rate":44100.0,"f0":7482.9927,"gain":0.0,"param":{"Q":1.414},"coeff":{"b0":0.7417508,"b1":-1.4835016,"b2":0.7417508,"a0":1.3095274,"a1":-0.96700317,"a2":0.6904726,"b0_div_a0":0.56642634,"b1_div_a0":-1.1328527,"b2_div_a0":0.56642634,"a1_div_a0":-0.73843676,"a2_div_a0":0.5272685}},"rhx":{"_ft":"FTBiquad","filter_type":"LowPass","rate":44100.0,"f0":5821.2,"gain":0.0,"param":{"Q":1.414},"coeff":{"b0":0.16233361,"b1":0.32466722,"b2":0.16233361,"a0":1.2607896,"a1":-1.3506656,"a2":0.73921037,"b0_div_a0":0.12875551,"b1_div_a0":0.25751102,"b2_div_a0":0.12875551,"a1_div_a0":-1.0712855,"a2_div_a0":0.58630747}}}]"#;
        let got = vec2ch_to_json(&json_to_vec2ch(t));
        assert_eq!(t, got);
    }
}
