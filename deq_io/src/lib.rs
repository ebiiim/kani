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
use std::io::prelude::*;
use std::process::{Command, Stdio};
use std::slice;
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
pub enum IO {
    Input,
    Output,
    Processor,
}

#[derive(Debug)]
pub enum Status {
    TxInit(IO),
    TxAck(IO),
    TxFin(IO),
    TxErr(IO),
    RxInit(IO),
    RxAck(IO),
    RxFin(IO),
    RxErr(IO),
    /// Latency in microseconds.
    ///
    /// Input and Output do not send latency for now,
    /// Processor sends filter latency (without IO latency).
    Latency(u32),
    /// Output inserted a frame to mitigate underrun.
    Interpolated(IO),
}

#[derive(Copy, Clone, Debug, Default)]
pub struct Info {
    pub frame: Frame,
    pub rate: Rate,
    pub input_ch: Ch,
    pub output_ch: Ch,
}

pub trait Input {
    fn info(&self) -> Info;
    fn run(
        &mut self,
        tx: SyncSender<Vec<f32>>,
        status_tx: SyncSender<Status>,
    ) -> Result<(), IOError>;
}

pub trait Output {
    fn info(&self) -> Info;
    fn run(&mut self, rx: Receiver<Vec<f32>>, status_tx: SyncSender<Status>)
        -> Result<(), IOError>;
}

pub trait Processor {
    fn info(&self) -> Info;
    fn run(
        &mut self,
        rx: Receiver<Vec<f32>>,
        tx: SyncSender<Vec<f32>>,
        status_tx: SyncSender<Status>,
    ) -> Result<(), IOError>;
}

#[derive(Debug)]
pub struct WaveReader {
    info: Info,
    path: String,
}

impl WaveReader {
    pub fn new(frame: Frame, path: &str) -> Result<Self, IOError> {
        let r = hound::WavReader::open(path);
        if let Err(e) = r {
            log::error!("could not open wav err={}", e);
            return Err(IOError::Format);
        }
        let r = r.unwrap();
        log::debug!("wav_info={:?}", r.spec());
        if r.spec().sample_format != hound::SampleFormat::Int
            || r.spec().bits_per_sample != 16
            || r.spec().channels != 2
        {
            log::error!("wav must be 2ch pcm_s16le for now");
            return Err(IOError::Format);
        }
        Ok(WaveReader {
            info: Info {
                frame,
                rate: r.spec().sample_rate as Rate,
                input_ch: 0,
                output_ch: r.spec().channels as Ch,
            },
            path: String::from(path),
        })
    }
    pub fn info(&self) -> Info {
        self.info
    }
}

impl Input for WaveReader {
    fn info(&self) -> Info {
        self.info
    }
    fn run(
        &mut self,
        tx: SyncSender<Vec<f32>>,
        status_tx: SyncSender<Status>,
    ) -> Result<(), IOError> {
        let reader = hound::WavReader::open(&self.path);
        if let Err(e) = reader {
            log::error!("could not open wav err={}", e);
            return Err(IOError::Format);
        }
        let mut reader = reader.unwrap();

        // TODO: read in play loop to reduce load time
        let buf: Vec<i16> = reader.samples().map(|s| s.unwrap()).collect();
        let ch = self.info.output_ch;
        let uframe = self.info.frame as usize * ch as usize;
        let mut buf = f::i16_to_f32(&buf);
        let zero_paddings = uframe - (buf.len() as usize % uframe);
        let zeros = vec![0.0; zero_paddings];
        buf.extend(zeros);
        assert_eq!(buf.len() % uframe, 0);
        let status_tx2 = status_tx.clone();
        let loopnum = buf.len() / uframe;

        // no return error after TxInit/RxInit
        status_tx.send(Status::TxInit(IO::Input)).unwrap();

        for i in 0..loopnum {
            let a = i * uframe;
            let b = (i + 1) * uframe;
            let buf = buf[a..b].to_vec();
            assert_eq!(buf.len(), uframe);
            match tx.send(buf) {
                Ok(_) => {
                    status_tx2.send(Status::TxAck(IO::Input)).unwrap();
                }
                Err(e) => {
                    status_tx2.send(Status::TxErr(IO::Input)).unwrap();
                    log::error!("{}", e);
                }
            }
        }

        status_tx2.send(Status::TxFin(IO::Input)).unwrap();
        Ok(())
    }
}

// #[derive(Debug)]
// pub struct WaveWriter {}

// impl WaveWriter {}

// impl Output for WaveWriter {
//     fn run(&self, rx: Receiver<Vec<f32>>, status_tx: SyncSender<Status>) -> Result<(), IOError> {
//         unimplemented!();
//     }
// }

#[derive(Debug, Default)]
pub struct PAReader {
    info: Info,
    dev: usize,
}

impl PAReader {
    pub fn new(dev: usize, frame: Frame, rate: Rate, ch: Ch) -> Self {
        PAReader {
            info: Info {
                frame,
                rate,
                input_ch: 0,
                output_ch: ch,
            },
            dev,
        }
    }
    pub fn info(&self) -> Info {
        self.info
    }
}

impl Input for PAReader {
    fn info(&self) -> Info {
        self.info
    }
    fn run(
        &mut self,
        tx: SyncSender<Vec<f32>>,
        status_tx: SyncSender<Status>,
    ) -> Result<(), IOError> {
        let pa = pa::PortAudio::new();
        if pa.is_err() {
            log::error!("could not initialize portaudio: {:?}", pa.unwrap_err());
            return Err(IOError::Device);
        }
        let pa = pa.unwrap();
        let ch = self.info.output_ch as i32;
        let interleaved = true;
        let frame = self.info.frame as u32;
        let player_info = get_device_info(&pa, self.dev)?;
        let rate = self.info.rate as f64;
        let latency = player_info.default_low_input_latency;
        let stream_params = pa::StreamParameters::<f32>::new(
            pa::DeviceIndex(self.dev as u32),
            ch,
            interleaved,
            latency,
        );
        log::debug!("use params={:?} rate={}", stream_params, rate);
        if let Err(e) = pa.is_input_format_supported(stream_params, rate) {
            log::error!("input format not supported err={}", e);
            return Err(IOError::Format);
        }
        let settings = pa::InputStreamSettings::new(stream_params, rate, frame);
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

        status_tx.send(Status::TxInit(IO::Input)).unwrap();

        loop {
            match stream.read(frame) {
                Ok(b) => match tx.send(b.to_vec()) {
                    Ok(_) => {
                        status_tx.send(Status::TxAck(IO::Input)).unwrap();
                    }
                    Err(e) => {
                        status_tx.send(Status::TxErr(IO::Input)).unwrap();
                        log::warn!("TxErr err={}", e);
                    }
                },
                Err(e) => {
                    status_tx.send(Status::TxErr(IO::Input)).unwrap();
                    log::warn!("TxErr err={}", e);
                }
            }
        }

        // TODO: how to stop this?
        if let Err(e) = stream.stop() {
            status_tx.send(Status::TxErr(IO::Input)).unwrap();
            log::warn!("could not stop input stream err={}", e);
        }

        status_tx.send(Status::TxFin(IO::Input)).unwrap();

        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct PAWriter {
    info: Info,
    dev: usize,
}

impl PAWriter {
    pub fn new(dev: usize, frame: Frame, rate: Rate, ch: Ch) -> Self {
        PAWriter {
            info: Info {
                frame,
                rate,
                input_ch: ch,
                output_ch: 0,
            },
            dev,
        }
    }
    pub fn info(&self) -> Info {
        self.info
    }
}

impl Output for PAWriter {
    fn info(&self) -> Info {
        self.info
    }
    fn run(
        &mut self,
        rx: Receiver<Vec<f32>>,
        status_tx: SyncSender<Status>,
    ) -> Result<(), IOError> {
        let pa = pa::PortAudio::new();
        if pa.is_err() {
            log::error!("could not initialize portaudio: {:?}", pa.unwrap_err());
            return Err(IOError::Device);
        }
        let pa = pa.unwrap();
        let ch = self.info.input_ch as i32;
        let interleaved = true;
        let frame = self.info.frame as u32;
        let player_info = get_device_info(&pa, self.dev)?;
        let rate = self.info.rate as f64;
        let latency = player_info.default_low_output_latency;
        let stream_params = pa::StreamParameters::<f32>::new(
            pa::DeviceIndex(self.dev as u32),
            ch,
            interleaved,
            latency,
        );
        log::debug!("use params={:?} rate={}", stream_params, rate);
        if let Err(e) = pa.is_output_format_supported(stream_params, rate) {
            log::error!("output format not supported err={}", e);
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

        status_tx.send(Status::RxInit(IO::Output)).unwrap();

        for buf in rx {
            let e = stream.write(frame, |w| {
                assert_eq!(buf.len(), num_samples);
                for (idx, sample) in buf.iter().enumerate() {
                    w[idx] = *sample;
                }
            });
            match e {
                Ok(_) => {
                    status_tx.send(Status::RxAck(IO::Output)).unwrap();
                }
                Err(e) => {
                    if format!("{}", e) == "OutputUnderflowed" {
                        // insert a same frame is audibly better than inserting a silent frame
                        stream
                            .write(frame, |w| {
                                assert_eq!(buf.len(), num_samples);
                                for (idx, sample) in buf.iter().enumerate() {
                                    w[idx] = *sample;
                                }
                                status_tx.send(Status::Interpolated(IO::Output)).unwrap();
                            })
                            .ok();
                    } else {
                        status_tx.send(Status::RxErr(IO::Output)).unwrap();
                        log::warn!("output stream err={}", e);
                    }
                }
            }
        }
        if let Err(e) = stream.stop() {
            status_tx.send(Status::RxErr(IO::Output)).unwrap();
            log::warn!("could not stop output stream err={}", e);
        }

        status_tx.send(Status::RxFin(IO::Output)).unwrap();

        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct PipeReader {
    info: Info,
    cmd: String,
    args: String,
}

impl PipeReader {
    pub fn new(cmd: &str, args: &str, frame: Frame, rate: Rate, ch: Ch) -> Self {
        PipeReader {
            info: Info {
                frame,
                rate,
                input_ch: 0,
                output_ch: ch,
            },
            cmd: String::from(cmd),
            args: String::from(args),
        }
    }
    pub fn info(&self) -> Info {
        self.info
    }
}

impl Input for PipeReader {
    fn info(&self) -> Info {
        self.info
    }
    fn run(
        &mut self,
        tx: SyncSender<Vec<f32>>,
        status_tx: SyncSender<Status>,
    ) -> Result<(), IOError> {
        let args: Vec<&str> = self.args.split(' ').collect();

        let mut process = match Command::new(&self.cmd)
            .args(args)
            .stdout(Stdio::piped())
            .spawn()
        {
            Err(e) => {
                log::error!("could not spawn process cmd={} err={}", &self.cmd, e);
                return Err(IOError::Device);
            }
            Ok(process) => process,
        };

        status_tx.send(Status::TxInit(IO::Input)).unwrap();

        let buflen = self.info.frame * self.info.output_ch * 2;
        let mut buf = vec![0; buflen];
        let mut o = process.stdout.take().unwrap();
        loop {
            match o.read(&mut buf) {
                Ok(n) => {
                    if n != buflen {
                        // Spotify Connect
                        // ignore all small data to avoid noise
                        // TODO: do this in e.g., SpotifyReader
                        continue;

                        // // ignore weird size inputs
                        // // ch x 2byte
                        // if n % self.info.output_ch * 2 != 0 {
                        //     log::warn!("ignored {} bytes of data from pipe", n);
                        //     continue;
                        // }
                        // // zero padding
                        // for idx in 0..(buflen - n) {
                        //     buf[buflen - idx - 1] = 0;
                        // }
                        // log::trace!("read {} bytes from pipe", n);
                        // let data = f::i16_to_f32(&reinterpret_u8_to_i16(&mut buf).to_vec());
                        // tx.send(data).unwrap();
                        // status_tx.send(Status::TxAck(IO::Input)).unwrap();

                        // // librespot closes pipe in every pause so no break here
                        // // TODO: no way to end
                        // // log::debug!("pipe ended");
                        // // break;
                    } else {
                        // buf.read
                        log::trace!("read {} bytes from pipe", n);
                        let data = f::i16_to_f32(&reinterpret_u8_to_i16(&mut buf).to_vec());
                        tx.send(data).unwrap();
                        status_tx.send(Status::TxAck(IO::Input)).unwrap();
                    };
                }
                Err(e) => {
                    status_tx.send(Status::TxErr(IO::Input)).unwrap();
                    log::debug!("read pipe err={:?}", e);
                }
            }
        }

        status_tx.send(Status::TxFin(IO::Input)).unwrap();

        Ok(())
    }
}

/// Note: I don't know how this works on big-endian machines.
fn reinterpret_u8_to_i16(b: &mut [u8]) -> &[i16] {
    assert_eq!(b.len() % 2, 0);
    unsafe { slice::from_raw_parts(b.as_mut_ptr() as *const i16, b.len() / 2) }
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

fn apply_filters(
    l: &[f32],
    r: &[f32],
    lfs: &mut f::VecFilters,
    rfs: &mut f::VecFilters,
) -> (Vec<f32>, Vec<f32>) {
    let l = lfs.iter_mut().fold(l.to_vec(), |x, f| f.apply(&x));
    let r = rfs.iter_mut().fold(r.to_vec(), |x, f| f.apply(&x));
    (l, r)
}

fn apply_filters2(l: &[f32], r: &[f32], sfs: &mut f::VecFilters2ch) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(l.len(), r.len());
    let debug = l.len();

    // this returns wrong length vecs
    // let (l, r) = sfs
    //     .iter_mut()
    //     .fold((l.to_vec(), r.to_vec()), |(l, r), f| f.apply(&l, &r));

    // compile error "destructuring assignments are unstable"
    // (l2, r2) = sfs[0].apply(&l, &r);

    let mut l = l.to_vec();
    let mut r = r.to_vec();
    for s in sfs {
        let (l2, r2) = s.apply(&l, &r);
        l = l2;
        r = r2;
    }

    assert_eq!(l.len(), r.len());
    assert_eq!(l.len(), debug);
    (l, r)
}

#[derive(Debug, Default)]
pub struct DSP {
    info: Info,
    lvf: f::VecFilters,
    rvf: f::VecFilters,
    vf2: f::VecFilters2ch,
}

impl DSP {
    pub fn new(frame: Frame, rate: Rate, vf2: f::VecFilters2ch) -> Self {
        let p = DSP {
            info: Info {
                frame,
                rate,
                input_ch: 2,
                output_ch: 2,
            },
            vf2,
            ..Default::default()
        };
        log::debug!("io::DSP {:?}", p);
        p
    }
    pub fn info(&self) -> Info {
        self.info
    }
}

impl Processor for DSP {
    fn info(&self) -> Info {
        self.info
    }
    fn run(
        &mut self,
        rx: Receiver<Vec<f32>>,
        tx: SyncSender<Vec<f32>>,
        status_tx: SyncSender<Status>,
    ) -> Result<(), IOError> {
        status_tx.send(Status::RxInit(IO::Processor)).unwrap();
        status_tx.send(Status::TxInit(IO::Processor)).unwrap();

        for buf in rx {
            status_tx.send(Status::RxAck(IO::Processor)).unwrap();
            let start = time::Instant::now();
            // --- measure latency ---
            let (l, r) = f::from_interleaved(&buf);
            let (l, r) = apply_filters(&l, &r, &mut self.lvf, &mut self.rvf);
            let (l, r) = apply_filters2(&l, &r, &mut self.vf2);
            let buf = f::to_interleaved(&l, &r);
            // -----------------------
            let end = start.elapsed();
            status_tx.send(Status::Latency(end.as_micros() as u32)).ok();
            match tx.send(buf) {
                Ok(_) => {
                    status_tx.send(Status::TxAck(IO::Processor)).unwrap();
                }
                Err(e) => {
                    status_tx.send(Status::TxErr(IO::Processor)).unwrap();
                    log::error!("could not send data {}", e);
                }
            }
        }
        status_tx.send(Status::RxFin(IO::Processor)).unwrap();
        status_tx.send(Status::TxFin(IO::Processor)).unwrap();
        Ok(())
    }
}
