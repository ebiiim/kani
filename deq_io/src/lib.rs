use deq_filter as f;
use deq_filter::Convolver;
use deq_filter::Delay;
use deq_filter::Filter;
use deq_filter::{BQFParam, BQFType, BiquadFilter};
use deq_filter::{VocalRemover, VocalRemoverType};
use deq_filter::{Volume, VolumeCurve};
use portaudio as pa;
use std::error;
use std::fmt;
use std::sync::mpsc::{Receiver, SyncSender};
use std::time;
extern crate deq_filter;

#[derive(Debug)]
pub enum IOError {
    Device,
    Format,
}

impl fmt::Display for IOError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let msg = match *self {
            IOError::Device => "device error",
            IOError::Format => "format error",
        };
        f.write_str(msg)
    }
}

impl error::Error for IOError {}

type Frame = usize;
type Rate = usize;
type Bit = usize;
type Ch = usize;

#[derive(Debug)]
pub enum Status {
    TxInit,
    TxAck,
    TxFin,
    TxErr,
    RxInit,
    RxAck,
    RxFin,
    RxErr,
    /// Latency in microseconds.
    ///
    /// Input and Output do not send latency for now,
    /// Processor sends filter latency (without IO latency).
    Latency(u32),
}

pub trait Input {
    fn run(&self, tx: SyncSender<Vec<f32>>, status_tx: SyncSender<Status>) -> Result<(), IOError>;
}

pub trait Output {
    fn run(&self, rx: Receiver<Vec<f32>>, status_tx: SyncSender<Status>) -> Result<(), IOError>;
}

pub trait Processor {
    fn run(
        &self,
        rx: Receiver<Vec<f32>>,
        tx: SyncSender<Vec<f32>>,
        status_tx: SyncSender<Status>,
    ) -> Result<(), IOError>;
}

#[derive(Debug)]
pub struct WaveReader {
    frame: Frame,
    path: String,
}

impl WaveReader {
    pub fn new(path: &str, frame: Frame) -> Self {
        // TODO: open reader and check spec here
        WaveReader {
            frame,
            path: String::from(path),
        }
    }
}

impl Input for WaveReader {
    fn run(&self, tx: SyncSender<Vec<f32>>, status_tx: SyncSender<Status>) -> Result<(), IOError> {
        let reader = hound::WavReader::open(&self.path);
        if let Err(e) = reader {
            log::error!("could not open wav err={}", e);
            return Err(IOError::Format);
        }
        let mut reader = reader.unwrap();
        log::debug!("wav_info={:?}", reader.spec());
        if reader.spec().sample_format != hound::SampleFormat::Int
            || reader.spec().bits_per_sample != 16
            || reader.spec().channels != 2
        {
            log::error!("wav must be 2ch pcm_s16le for now");
            return Err(IOError::Format);
        }
        // TODO: read in play loop?
        let buf: Vec<i16> = reader.samples().map(|s| s.unwrap()).collect();
        let ch = 2;
        let uframe = self.frame as usize * ch as usize;
        let mut buf = f::i16_to_f32(&buf);
        let zero_paddings = uframe - (buf.len() as usize % uframe);
        let zeros = vec![0.0; zero_paddings];
        buf.extend(zeros);
        assert_eq!(buf.len() % uframe, 0);
        let status_tx2 = status_tx.clone();
        let loopnum = buf.len() / uframe;

        status_tx.send(Status::TxInit).unwrap();

        for i in 0..loopnum {
            let a = i * uframe;
            let b = (i + 1) * uframe;
            let buf = buf[a..b].to_vec();
            assert_eq!(buf.len(), uframe);
            match tx.send(buf) {
                Ok(_) => {
                    status_tx2.send(Status::TxAck).unwrap();
                }
                Err(e) => {
                    status_tx2.send(Status::TxErr).unwrap();
                    log::error!("{}", e);
                }
            }
        }

        status_tx2.send(Status::TxFin).unwrap();
        Ok(())
    }
}

#[derive(Debug)]
pub struct WaveWriter {}

impl WaveWriter {}

impl Output for WaveWriter {
    fn run(&self, rx: Receiver<Vec<f32>>, status_tx: SyncSender<Status>) -> Result<(), IOError> {
        unimplemented!();
    }
}

#[derive(Debug)]
pub struct PAReader {}

impl PAReader {}

impl Input for PAReader {
    fn run(&self, tx: SyncSender<Vec<f32>>, status_tx: SyncSender<Status>) -> Result<(), IOError> {
        unimplemented!();
    }
}

#[derive(Debug)]
pub struct PAWriter {
    dev: usize,
    frame: Frame,
    rate: Rate,
    ch: Ch,
}

impl PAWriter {
    pub fn new(dev: usize, frame: Frame, rate: Rate, ch: Ch) -> Self {
        PAWriter {
            dev,
            frame,
            rate,
            ch,
        }
    }
}

impl Output for PAWriter {
    fn run(&self, rx: Receiver<Vec<f32>>, status_tx: SyncSender<Status>) -> Result<(), IOError> {
        let pa = pa::PortAudio::new();
        if pa.is_err() {
            log::error!("could not initialize portaudio: {:?}", pa.unwrap_err());
            return Err(IOError::Device);
        }
        let pa = pa.unwrap();
        let ch = 2;
        let interleaved = true;
        let frame = self.frame as u32;
        let player_info = get_device_info(&pa, self.dev)?;
        let rate = self.rate as f64;
        let latency = player_info.default_low_output_latency;
        let stream_params = pa::StreamParameters::<f32>::new(
            pa::DeviceIndex(self.dev as u32),
            ch,
            interleaved,
            latency,
        );
        if let Err(e) = pa.is_output_format_supported(stream_params, rate) {
            log::error!("format not supported err={}", e);
            return Err(IOError::Format);
        }
        let settings = pa::OutputStreamSettings::new(stream_params, rate, frame);
        let stream = pa.open_blocking_stream(settings);
        if let Err(e) = stream {
            log::error!("could not open stream err={}", e);
            return Err(IOError::Format);
        }
        let mut stream = stream.unwrap();
        if let Err(e) = stream.start() {
            log::error!("could not start stream err={}", e);
            return Err(IOError::Format);
        }
        log::debug!("stream started info={:?}", stream.info());
        let num_samples = (frame * ch as u32) as usize;

        status_tx.send(Status::RxInit).unwrap();

        for buf in rx {
            if let Err(e) = stream.write(frame, |w| {
                assert_eq!(buf.len(), num_samples);
                for (idx, sample) in buf.iter().enumerate() {
                    w[idx] = *sample;
                }
            }) {
                status_tx.send(Status::RxErr).unwrap();
                log::warn!("playback stream err={}", e);
            };
        }

        if let Err(e) = stream.stop() {
            status_tx.send(Status::RxErr).unwrap();
            log::warn!("could not stop playback stream err={}", e);
        }

        status_tx.send(Status::RxFin).unwrap();

        Ok(())
    }
}

pub fn get_device_names(pa: &pa::PortAudio) -> Result<Vec<(usize, String)>, IOError> {
    let devs = pa.devices();
    match devs {
        Ok(devs) => Ok(devs
            .map(|dev| {
                let d = dev.unwrap();
                let pa::DeviceIndex(idx) = d.0;
                (idx as usize, String::from(d.1.name))
            })
            .collect()),
        Err(_) => Err(IOError::Device),
    }
}

pub fn get_device_info(pa: &pa::PortAudio, idx: usize) -> Result<pa::DeviceInfo, IOError> {
    let devidx = pa::DeviceIndex(idx as u32);
    let devs = pa.devices();
    if let Ok(devs) = devs {
        for d in devs {
            if let Ok(d) = d {
                if d.0 == devidx {
                    return Ok(d.1);
                }
            } else {
                return Err(IOError::Device);
            }
        }
    }
    Err(IOError::Device)
}

fn setup_filters(fs: f32) -> (Vec<Box<dyn f::Filter>>, Vec<Box<dyn f::Filter>>) {
    // init filters
    let lfs: Vec<Box<dyn f::Filter>> = vec![
        // BiquadFilter::newb(BQFType::HighPass, fs, 250.0, 0.0, BQFParam::Q(0.707)),
        // BiquadFilter::newb(BQFType::LowPass, fs, 8000.0, 0.0, BQFParam::Q(0.707)),
        // BiquadFilter::newb(BQFType::PeakingEQ, fs, 880.0, 9.0, BQFParam::BW(1.0)),
        Volume::newb(VolumeCurve::Gain, -12.0),
        // Delay::newb(200, fs as usize),
    ];
    let rfs: Vec<Box<dyn f::Filter>> = vec![
        // BiquadFilter::newb(BQFType::HighPass, fs, 250.0, 0.0, BQFParam::Q(0.707)),
        // BiquadFilter::newb(BQFType::LowPass, fs, 8000.0, 0.0, BQFParam::Q(0.707)),
        // BiquadFilter::newb(BQFType::PeakingEQ, fs, 880.0, 9.0, BQFParam::BW(1.0)),
        Volume::newb(VolumeCurve::Gain, -12.0),
        // Delay::newb(200, fs as usize),
    ];
    (lfs, rfs)
}

fn setup_filters2(fs: f32) -> Vec<Box<dyn f::Filter2ch>> {
    let sfs: Vec<Box<dyn f::Filter2ch>> = vec![
        // VocalRemover::newb(VocalRemoverType::RemoveCenter),
        // VocalRemover::newb(VocalRemoverType::RemoveCenterBW(fs, f32::MIN, f32::MAX)),
        VocalRemover::newb(VocalRemoverType::RemoveCenterBW(fs, 200.0, 4800.0)),
    ];
    sfs
}

fn apply_filters(
    l: &[f32],
    r: &[f32],
    lfs: &mut Vec<Box<dyn f::Filter>>,
    rfs: &mut Vec<Box<dyn f::Filter>>,
) -> (Vec<f32>, Vec<f32>) {
    let l = lfs.iter_mut().fold(l.to_vec(), |x, f| f.apply(&x));
    let r = rfs.iter_mut().fold(r.to_vec(), |x, f| f.apply(&x));
    (l, r)
}

fn apply_filters2(
    l: &[f32],
    r: &[f32],
    sfs: &mut Vec<Box<dyn f::Filter2ch>>,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(l.len(), r.len());
    let debug = l.len();

    // this returns wrong length vecs
    // let (l, r) = sfs
    //     .iter_mut()
    //     .fold((l.to_vec(), r.to_vec()), |(l, r), f| f.apply(&l, &r));

    let mut l = l.to_vec();
    let mut r = r.to_vec();
    // compile error "destructuring assignments are unstable"
    // (l2, r2) = sfs[0].apply(&l, &r);
    let (l2, r2) = sfs[0].apply(&l, &r);
    l = l2;
    r = r2;

    assert_eq!(l.len(), r.len());
    assert_eq!(l.len(), debug);
    (l, r)
}

pub struct DSP {}

impl Processor for DSP {
    fn run(
        &self,
        rx: Receiver<Vec<f32>>,
        tx: SyncSender<Vec<f32>>,
        status_tx: SyncSender<Status>,
    ) -> Result<(), IOError> {
        // init filters
        let fs = 48000.0;
        let (mut lfs, mut rfs) = setup_filters(fs);
        let mut sfs = setup_filters2(fs);

        status_tx.send(Status::RxInit).unwrap();
        status_tx.send(Status::TxInit).unwrap();

        for buf in rx {
            status_tx.send(Status::RxAck).unwrap();
            let start = time::Instant::now();
            // --- measure latency ---
            let (l, r) = f::from_interleaved(&buf);
            let (l, r) = apply_filters(&l, &r, &mut lfs, &mut rfs);
            let (l, r) = apply_filters2(&l, &r, &mut sfs);
            let buf = f::to_interleaved(&l, &r);
            // -----------------------
            let end = start.elapsed();
            status_tx.send(Status::Latency(end.as_micros() as u32)).ok();
            match tx.send(buf) {
                Ok(_) => {
                    status_tx.send(Status::TxAck).unwrap();
                }
                Err(e) => {
                    status_tx.send(Status::TxErr).unwrap();
                    log::error!("could not send data {}", e);
                }
            }
        }
        status_tx.send(Status::RxFin).unwrap();
        status_tx.send(Status::TxFin).unwrap();
        Ok(())
    }
}
