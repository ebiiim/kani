pub fn i16_to_f32(a: Vec<i16>) -> Vec<f32> {
    log::trace!("i16_to_f32 a.len()={}", a.len());
    a.iter().map(|x| *x as f32 / 32767.0).collect()
}

pub fn from_interleaved(a: Vec<f32>) -> (Vec<f32>, Vec<f32>) {
    log::trace!("from_interleaved a.len()={}", a.len());
    let mut l: Vec<f32> = Vec::with_capacity(a.len() / 2);
    let mut r: Vec<f32> = Vec::with_capacity(a.len() / 2);
    for (i, s) in a.iter().enumerate() {
        if i % 2 == 0 {
            l.push(*s);
        } else {
            r.push(*s);
        }
    }
    log::trace!("from_interleaved l.len()={} r.len()={}", l.len(), r.len());
    (l, r)
}

pub fn to_interleaved(l: Vec<f32>, r: Vec<f32>) -> Vec<f32> {
    log::trace!("to_interleaved l.len()={} r.len()={}", l.len(), r.len());
    if l.len() != r.len() {
        log::error!(
            "channel buffers must have same length but l={} r={}",
            l.len(),
            r.len()
        );
        return Vec::new();
    }
    let mut a: Vec<f32> = Vec::with_capacity(l.len() * 2);
    for i in 0..(l.len() * 2) {
        if i % 2 == 0 {
            a.push(l[i / 2]);
        } else {
            a.push(r[i / 2]);
        }
    }
    log::trace!("to_interleaved a.len()={}", a.len());
    a
}

pub fn apply<T: Appliable>(mut filter: T, samples: Vec<f32>) -> Vec<f32> {
    filter.apply(samples)
}

pub trait Appliable {
    fn apply(&mut self, samples: Vec<f32>) -> Vec<f32>;
}

#[derive(Debug)]
pub struct Volume {
    curve: VolumeCurve,
    vol: f64,
}

#[derive(Debug)]
pub enum VolumeCurve {
    Liner,
}

impl Volume {
    pub fn new(curve: VolumeCurve, vol: f64) -> Self {
        log::debug!("volume curve={:?} vol={}", curve, vol);
        Self { curve, vol }
    }
}

impl Appliable for Volume {
    fn apply(&mut self, samples: Vec<f32>) -> Vec<f32> {
        match self.curve {
            VolumeCurve::Liner => samples.iter().map(|x| *x * self.vol as f32).collect(),
        }
    }
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
    Q(f64),
    // Bandwidth (Octave)
    BW(f64),
    // Slope
    S(f64),
}

#[derive(Debug)]
struct BQFCoeff {
    b0: f64,
    b1: f64,
    b2: f64,
    a0: f64,
    a1: f64,
    a2: f64,
    b0_div_a0: f64,
    b1_div_a0: f64,
    b2_div_a0: f64,
    a1_div_a0: f64,
    a2_div_a0: f64,
}

impl BQFCoeff {
    pub fn new(filter_type: &BQFType, rate: f64, f0: f64, gain: f64, param: &BQFParam) -> BQFCoeff {
        let a = 10.0f64.powf(gain / 40.0);
        let w0 = 2.0 * std::f64::consts::PI * f0 / rate;
        let w0_cos = w0.cos();
        let w0_sin = w0.sin();
        let alpha = match param {
            BQFParam::Q(q) => w0_sin / (2.0 * q),
            BQFParam::BW(bw) => {
                // I'm not so sure about this
                // w0_sin * (2.0f64.log10() / 2.0 * bw * w0 / w0_sin).sinh()
                // So I'd like to convert Bandwidth (octave) to Q based on formula [here](http://www.sengpielaudio.com/calculator-bandwidth.htm)
                let bw2q = |bw: f64| (2.0f64.powf(bw)).sqrt() / (2.0f64.powf(bw) - 1.0);
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
}

#[derive(Debug)]
pub struct BiquadFilter {
    /// filter type
    filter_type: BQFType,
    /// sampling rate
    rate: f64,
    /// cutoff/center freq
    f0: f64,
    /// gain in dB (PeakingEQ | LowShelf | HighShelf)     
    gain: f64,
    /// Q/Bandwidth/Slope value
    param: BQFParam,
    /// input delay buffer
    buf_x: [f64; 2],
    /// output delay buffer
    buf_y: [f64; 2],
    /// coefficients
    coeff: BQFCoeff,
}

impl BiquadFilter {
    pub fn new(filter_type: BQFType, rate: f64, f0: f64, gain: f64, param: BQFParam) -> Self {
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
}

impl Appliable for BiquadFilter {
    fn apply(&mut self, samples: Vec<f32>) -> Vec<f32> {
        let mut buf: Vec<f32> = Vec::with_capacity(samples.len());
        for x in samples.iter() {
            let x = *x as f64;
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
            buf.push(y as f32);
        }
        buf
    }
}
