use anyhow::{bail, Context, Result};
use kani_filter::*;
use portaudio as pa;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::io::prelude::*;
use std::process::{Command, Stdio};
use std::slice;
use std::sync::mpsc::{Receiver, SyncSender};
use std::time;

extern crate kani_filter;

type Frame = usize;
type Rate = usize;
type Bit = usize;
type Ch = usize;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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
    /// Currently only Processor sends this value.
    /// Processor sends signal process latency (without IO latency).
    Latency(u32),
    /// Output inserted a frame to mitigate underrun.
    Interpolated(IO),
    /// RMS in FP32, you might want to calc gain.
    ///
    /// Processor sends this value and `IO` here means where the value is measured.
    /// Window size depends on Processor implementation.
    RMS(IO, Ch, f32),
    /// Peak in FP32, you might want to calc gain.
    ///
    /// Processor sends this value and `IO` here means where the value is measured.
    /// Detection method depends on Processor implementation.
    Peak(IO, Ch, f32),
}

#[derive(Debug)]
pub enum Cmd {
    /// Send config and reload
    Reload(String),
    // TODO: Start, Stop, Pause?
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
        cmd_rx: Receiver<Cmd>,
    ) -> Result<()>;
}

pub trait Output {
    fn info(&self) -> Info;
    fn run(
        &mut self,
        rx: Receiver<Vec<f32>>,
        status_tx: SyncSender<Status>,
        cmd_rx: Receiver<Cmd>,
    ) -> Result<()>;
}

pub trait Processor {
    fn info(&self) -> Info;
    fn run(
        &mut self,
        rx: Receiver<Vec<f32>>,
        tx: SyncSender<Vec<f32>>,
        status_tx: SyncSender<Status>,
        cmd_rx: Receiver<Cmd>,
    ) -> Result<()>;
}

pub struct WaveReader {
    info: Info,
    path: String,
    reader: hound::WavReader<std::io::BufReader<std::fs::File>>,
}

impl Debug for WaveReader {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "io::WaveReader info={:?} path={}",
            self.info(),
            self.path
        )
    }
}

impl WaveReader {
    pub fn new(frame: Frame, path: &str) -> Result<Self> {
        let r = hound::WavReader::open(path)
            .with_context(|| format!("could not open wav path={}", path))?;
        log::debug!("wav_info={:?}", r.spec());
        if r.spec().sample_format != hound::SampleFormat::Int
            || r.spec().bits_per_sample != 16
            || r.spec().channels != 2
        {
            bail!("currenlty wav must be 2ch pcm_s16le");
        }
        Ok(WaveReader {
            info: Info {
                frame,
                rate: r.spec().sample_rate as Rate,
                input_ch: 0,
                output_ch: r.spec().channels as Ch,
            },
            reader: r,
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
        cmd_rx: Receiver<Cmd>,
    ) -> Result<()> {
        // samples per frame
        let spf = self.info.frame as usize * self.info.output_ch as usize;
        // no return error after TxInit/RxInit
        status_tx.send(Status::TxInit(IO::Input)).unwrap();
        loop {
            let mut finished = false;
            let mut buf: Vec<i16> = Vec::with_capacity(spf);
            for _ in 0..spf {
                match self.reader.samples::<i16>().next() {
                    Some(v) => buf.push(v.unwrap()),
                    None => {
                        buf.push(0);
                        finished = true;
                    }
                }
            }
            assert_eq!(buf.len(), spf);
            let buf = i16_to_f32(&buf);
            match tx.send(buf) {
                Ok(_) => {
                    status_tx.send(Status::TxAck(IO::Input)).unwrap();
                }
                Err(e) => {
                    status_tx.send(Status::TxErr(IO::Input)).unwrap();
                    log::error!("{}", e);
                }
            }
            if finished {
                break;
            }
        }
        status_tx.send(Status::TxFin(IO::Input)).unwrap();
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
        cmd_rx: Receiver<Cmd>,
    ) -> Result<()> {
        let pa = pa::PortAudio::new().with_context(|| "could not initialize portaudio")?;
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
        pa.is_input_format_supported(stream_params, rate)
            .with_context(|| "input format not supported")?;
        let settings = pa::InputStreamSettings::new(stream_params, rate, frame);
        let mut stream = pa
            .open_blocking_stream(settings)
            .with_context(|| "could not open stream")?;
        stream.start().with_context(|| "could not start stream")?;
        log::debug!("stream started info={:?}", stream.info());

        status_tx.send(Status::TxInit(IO::Input)).unwrap();

        loop {
            // poll commands
            let cmd = cmd_rx.try_recv();
            if cmd.is_ok() {
                // do nothing
                log::trace!("PAReader cmd_rx={:?}", cmd);
            }
            // process the frame
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
        cmd_rx: Receiver<Cmd>,
    ) -> Result<()> {
        let pa = pa::PortAudio::new().with_context(|| "could not initialize portaudio")?;
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
        pa.is_output_format_supported(stream_params, rate)
            .with_context(|| "output format not supported")?;
        let settings = pa::OutputStreamSettings::new(stream_params, rate, frame);
        let mut stream = pa
            .open_blocking_stream(settings)
            .with_context(|| "could not open stream")?;
        stream.start().with_context(|| "could not start stream")?;
        log::debug!("stream started info={:?}", stream.info());
        let num_samples = (frame * ch as u32) as usize;

        status_tx.send(Status::RxInit(IO::Output)).unwrap();

        for buf in rx {
            // poll commands
            let cmd = cmd_rx.try_recv();
            if cmd.is_ok() {
                // do nothing
                log::trace!("PAWriter cmd_rx={:?}", cmd);
            }
            // process the frame
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
        cmd_rx: Receiver<Cmd>,
    ) -> Result<()> {
        let args: Vec<&str> = self.args.split(' ').collect();

        let mut process = Command::new(&self.cmd)
            .args(args)
            .stdout(Stdio::piped())
            .spawn()
            .with_context(|| {
                format!(
                    "could not spawn process cmd={} args={}",
                    self.cmd, self.args
                )
            })?;

        status_tx.send(Status::TxInit(IO::Input)).unwrap();

        let buflen = self.info.frame * self.info.output_ch * 2;
        let mut buf = vec![0; buflen];
        let mut o = process.stdout.take().unwrap();
        loop {
            // poll commands
            let cmd = cmd_rx.try_recv();
            if cmd.is_ok() {
                // do nothing
                log::trace!("PipeReader cmd_rx={:?}", cmd);
            }
            // process the frame
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
                        // let data = i16_to_f32(&reinterpret_u8_to_i16(&mut buf).to_vec());
                        // tx.send(data).unwrap();
                        // status_tx.send(Status::TxAck(IO::Input)).unwrap();

                        // // librespot closes pipe in every pause so no break here
                        // // TODO: no way to end
                        // // log::debug!("pipe ended");
                        // // break;
                    } else {
                        // buf.read
                        log::trace!("read {} bytes from pipe", n);
                        let data = i16_to_f32(&reinterpret_u8_to_i16(&mut buf).to_vec());
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

pub fn get_device_names(pa: &pa::PortAudio) -> Result<Vec<(usize, String)>> {
    let devs = pa.devices().with_context(|| "could not fetch devices")?;
    Ok(devs
        .map(|dev| {
            let d = dev.unwrap();
            let pa::DeviceIndex(idx) = d.0;
            (idx as usize, String::from(d.1.name))
        })
        .collect())
}

pub fn get_device_info(pa: &pa::PortAudio, idx: usize) -> Result<pa::DeviceInfo> {
    let devidx = pa::DeviceIndex(idx as u32);
    let devs = pa.devices().with_context(|| "could not fetch devices")?;
    for (i, d) in devs.enumerate() {
        let d = d.with_context(|| format!("could not get device {}", i))?;
        if d.0 == devidx {
            return Ok(d.1);
        }
    }
    bail!("device {} not found", idx);
}

fn apply_filters(
    l: &[f32],
    r: &[f32],
    lfs: &mut VecFilters,
    rfs: &mut VecFilters,
) -> (Vec<f32>, Vec<f32>) {
    let l = lfs.iter_mut().fold(l.to_vec(), |x, f| f.apply(&x));
    let r = rfs.iter_mut().fold(r.to_vec(), |x, f| f.apply(&x));
    (l, r)
}

fn apply_filters2(l: &[f32], r: &[f32], sfs: &mut VecFilters2ch) -> (Vec<f32>, Vec<f32>) {
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
    lvf: VecFilters,
    rvf: VecFilters,
    vf2: VecFilters2ch,
}

impl DSP {
    pub fn new(frame: Frame, rate: Rate, vf2_json: &str) -> Self {
        let vf2 = parse_vec2ch(vf2_json, rate as f32);
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
        cmd_rx: Receiver<Cmd>,
    ) -> Result<()> {
        // RMS window = 100 ms
        // peak hold timer = 2000 ms
        let mut l_in_level = LevelMeter1ch::new(
            status_tx.clone(),
            IO::Input,
            0,
            (self.info.rate as f32 * 0.1) as usize,
            self.info.rate * 2,
        );
        let mut r_in_level = LevelMeter1ch::new(
            status_tx.clone(),
            IO::Input,
            1,
            (self.info.rate as f32 * 0.1) as usize,
            self.info.rate * 2,
        );
        let mut l_out_level = LevelMeter1ch::new(
            status_tx.clone(),
            IO::Output,
            0,
            (self.info.rate as f32 * 0.1) as usize,
            self.info.rate * 2,
        );
        let mut r_out_level = LevelMeter1ch::new(
            status_tx.clone(),
            IO::Output,
            1,
            (self.info.rate as f32 * 0.1) as usize,
            self.info.rate * 2,
        );

        status_tx.send(Status::RxInit(IO::Processor)).unwrap();
        status_tx.send(Status::TxInit(IO::Processor)).unwrap();

        for buf in rx {
            // poll commands
            let cmd = cmd_rx.try_recv();
            if cmd.is_ok() {
                log::trace!("DSP cmd_rx={:?}", cmd);
                match cmd.unwrap() {
                    Cmd::Reload(s) => {
                        log::debug!("DSP reload config={}", s);
                        self.vf2 = parse_vec2ch(&s, self.info.rate as f32);
                    }
                }
            }
            // process the frame
            status_tx.send(Status::RxAck(IO::Processor)).unwrap();
            let start = time::Instant::now();
            // --- measure latency ---
            let (l, r) = from_interleaved(&buf);
            l_in_level.push_data(&l);
            r_in_level.push_data(&r);
            let (l, r) = apply_filters(&l, &r, &mut self.lvf, &mut self.rvf);
            let (l, r) = apply_filters2(&l, &r, &mut self.vf2);
            l_out_level.push_data(&l);
            r_out_level.push_data(&r);
            let buf = to_interleaved(&l, &r);
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

#[derive(Debug)]
struct LevelMeter1ch {
    status_tx: SyncSender<Status>,
    io: IO,
    ch: Ch,
    /// current mean square (calc root before every send)
    rms_cur: f32,
    rms_cnt: u64,
    rms_window: Frame,
    rms_buf: VecDeque<f32>,
    peak_cur: f32,
    peak_cnt: u64,
    peak_hold: Frame,
}

impl LevelMeter1ch {
    pub fn new(
        status_tx: SyncSender<Status>,
        io: IO,
        ch: Ch,
        rms_window: Frame,
        peak_hold: Frame,
    ) -> Self {
        Self {
            status_tx,
            io,
            ch,
            rms_cur: 0.0,
            rms_cnt: 0,
            rms_window,
            rms_buf: VecDeque::from(vec![0.0; rms_window]),
            peak_cur: 0.0,
            peak_cnt: 0,
            peak_hold: peak_hold,
        }
    }
    pub fn push_data(&mut self, s: &[f32]) {
        for i in 0..s.len() {
            let oldest_sample = self.rms_buf.pop_front().unwrap();
            let latest_sample = s[i];
            self.rms_buf.push_back(latest_sample);

            // calc RMS
            if self.rms_cnt % self.rms_window as u64 == 0 {
                self.rms_cur = self.rms_cur - oldest_sample.powi(2) + latest_sample.powi(2);
                self.status_tx
                    .send(Status::RMS(self.io, self.ch, self.rms_cur.sqrt()))
                    .unwrap();
            }
            self.rms_cnt += 1;

            // check peak
            let latest_abs = latest_sample.abs();
            if self.peak_cur < latest_abs || self.peak_cnt % self.peak_hold as u64 == 0 {
                self.peak_cur = latest_abs;
                self.status_tx
                    .send(Status::Peak(self.io, self.ch, self.peak_cur))
                    .unwrap();
                self.peak_cnt = 0; // reset peak hold timer
            }
            self.peak_cnt += 1
        }
    }
}

fn parse_vec2ch(json: &str, fs: f32) -> VecFilters2ch {
    match json_to_vec2ch(json, fs) {
        Ok(vf2) => vf2,
        Err(_) => {
            let default_filters = setup_vf2(fs);
            log::warn!(
                "could not parse json so default config loaded: {}",
                vec2ch_to_json(&default_filters)
            );
            default_filters
        }
    }
}

fn setup_vf(fs: f32) -> (VecFilters, VecFilters) {
    let lvf: VecFilters = vec![
        // BiquadFilter::newb(BQFType::HighPass, fs, 250.0, 0.0, BQFParam::Q(0.707)),
        // BiquadFilter::newb(BQFType::LowPass, fs, 8000.0, 0.0, BQFParam::Q(0.707)),
        // BiquadFilter::newb(BQFType::PeakingEQ, fs, 880.0, 9.0, BQFParam::BW(1.0)),
        // Volume::newb(VolumeCurve::Gain, -6.0),
        // Delay::newb(200, fs as usize),
        NopFilter::newb(),
    ];
    let rvf: VecFilters = vec![
        // BiquadFilter::newb(BQFType::HighPass, fs, 250.0, 0.0, BQFParam::Q(0.707)),
        // BiquadFilter::newb(BQFType::LowPass, fs, 8000.0, 0.0, BQFParam::Q(0.707)),
        // BiquadFilter::newb(BQFType::PeakingEQ, fs, 880.0, 9.0, BQFParam::BW(1.0)),
        // Volume::newb(VolumeCurve::Gain, -6.0),
        // Delay::newb(200, fs as usize),
        NopFilter::newb(),
    ];
    log::info!("filters (L ch): {}", vec_to_json(&lvf));
    log::info!("filters (R ch): {}", vec_to_json(&rvf));
    (lvf, rvf)
}

fn setup_vf2(fs: f32) -> VecFilters2ch {
    let pf = PairedFilter::newb(
        // NopFilter::newb(),
        // NopFilter::newb(),
        Volume::newb(VolumeCurve::Gain, -6.0),
        Volume::newb(VolumeCurve::Gain, -6.0),
        fs,
    );
    let vf2: VecFilters2ch = vec![
        pf,
        // VocalRemover::newb(VocalRemoverType::RemoveCenter),
        // VocalRemover::newb(VocalRemoverType::RemoveCenterBW(fs, f32::MIN, f32::MAX)),
        VocalRemover::newb(VocalRemoverType::RemoveCenterBW(240.0, 4400.0), fs),
    ];
    log::info!("filters (L&R ch): {}", vec2ch_to_json(&vf2));
    vf2
}
