use deq_filter as f;
use portaudio as pa;
use std::env;
use std::error;
use std::fmt;
use std::process;
use std::thread;

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

use std::sync::mpsc::{channel, Receiver, Sender};

pub trait Input {
    fn start(&self) -> Result<Receiver<Vec<f32>>, IOError>;
}

pub trait Output {
    fn start(&self, rx: Receiver<Vec<f32>>) -> Result<(), IOError>;
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
    pub fn newb(path: &str, frame: Frame) -> Box<dyn Input> {
        Box::new(Self::new(path, frame))
    }
}

impl Input for WaveReader {
    fn start(&self) -> Result<Receiver<Vec<f32>>, IOError> {
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
            log::debug!("wav must be 2ch pcm_s16le");
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

        let (tx, rx) = channel();

        let loopnum = buf.len() / uframe;
        let _ = thread::spawn(move || {
            for i in 0..loopnum {
                let a = i * uframe;
                let b = (i + 1) * uframe;
                let buf = buf[a..b].to_vec();
                assert_eq!(buf.len(), uframe);
                tx.send(buf).unwrap();
            }
            drop(tx);
        });

        Ok(rx)
    }
}

#[derive(Debug)]
pub struct WaveWriter {}

impl WaveWriter {}

impl Output for WaveWriter {
    fn start(&self, rx: Receiver<Vec<f32>>) -> Result<(), IOError> {
        unimplemented!();
    }
}

#[derive(Debug)]
pub struct PAReader {}

impl PAReader {}

impl Input for PAReader {
    fn start(&self) -> Result<Receiver<Vec<f32>>, IOError> {
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
    pub fn newb(dev: usize, frame: Frame, rate: Rate, ch: Ch) -> Box<dyn Output> {
        Box::new(Self::new(dev, frame, rate, ch))
    }
}

impl Output for PAWriter {
    fn start(&self, rx: Receiver<Vec<f32>>) -> Result<(), IOError> {
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
        let stream_settings = pa::OutputStreamSettings::new(stream_params, rate, frame);
        log::trace!("open stream");
        let stream = pa.open_blocking_stream(stream_settings);
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
        // playback
        let num_samples = (frame * ch as u32) as usize;
        for buf in rx {
            if let Err(e) = stream.write(frame, |w| {
                assert_eq!(buf.len(), num_samples);
                for (idx, sample) in buf.iter().enumerate() {
                    w[idx] = *sample;
                }
            }) {
                log::warn!("playback stream err={}", e);
            };
        }
        if let Err(e) = stream.stop() {
            log::warn!("could not stop playback stream err={}", e);
        }
        Ok(())
    }
}

pub fn get_device_names(pa: &pa::PortAudio) -> Result<Vec<(usize, String)>, IOError> {
    log::trace!("get devices");
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
    log::trace!("get devices");
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
