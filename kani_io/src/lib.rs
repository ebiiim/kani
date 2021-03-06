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
    /// RMS in magnitude (abs of signal value), you might want to calc gain.
    ///
    /// Processor sends this value and `IO` here means where the value is measured.
    /// Window size depends on Processor implementation.
    RMS(IO, Ch, S),
    /// Peak in magnitude (abs of signal value), you might want to calc gain.
    ///
    /// Processor sends this value and `IO` here means where the value is measured.
    /// Detection method depends on Processor implementation.
    Peak(IO, Ch, S),
    /// Processor sends this value when reloaded config.
    Loaded(String),
}

#[derive(Debug)]
pub enum Cmd {
    /// Processor reloads config when received this Cmd.
    Reload(String),
    /// Input, Output and Processor wait this Cmd after sending TxInit/RxInit for synchronization.
    Start,
    /// Input stops when received this Cmd,
    /// and then Processor & Output also stop as the tx channel close.
    Stop,
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Info {
    pub input_frame: Frame,
    pub input_rate: Hz,
    pub input_ch: Ch,

    pub output_frame: Frame,
    pub output_rate: Hz,
    pub output_ch: Ch,
}

pub trait Input {
    fn info(&self) -> &Info;
    fn run(
        &mut self,
        tx: SyncSender<Vec<S>>,
        status_tx: SyncSender<Status>,
        cmd_rx: Receiver<Cmd>,
    ) -> Result<()>;
}

pub trait Output {
    fn info(&self) -> &Info;
    fn run(
        &mut self,
        rx: Receiver<Vec<S>>,
        status_tx: SyncSender<Status>,
        cmd_rx: Receiver<Cmd>,
    ) -> Result<()>;
}

pub trait Processor {
    fn info(&self) -> &Info;
    fn run(
        &mut self,
        rx: Receiver<Vec<S>>,
        tx: SyncSender<Vec<S>>,
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
                input_frame: 0,
                input_rate: 0,
                input_ch: 0,

                output_frame: frame,
                output_rate: r.spec().sample_rate as Hz,
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
    fn info(&self) -> &Info {
        &self.info
    }
    fn run(
        &mut self,
        tx: SyncSender<Vec<S>>,
        status_tx: SyncSender<Status>,
        cmd_rx: Receiver<Cmd>,
    ) -> Result<()> {
        // samples per frame
        let spf = self.info.output_frame as usize * self.info.output_ch as usize;
        // no return error after TxInit/RxInit
        status_tx.send(Status::TxInit(IO::Input)).unwrap();
        // wait Cmd::Start for synchronization (currently no check)
        cmd_rx.recv().unwrap();
        loop {
            // poll commands
            let cmd = cmd_rx.try_recv();
            if cmd.is_ok() {
                match cmd.unwrap() {
                    Cmd::Stop => {
                        log::debug!("WaveReader received Cmd::Stop so stop now");
                        break;
                    }
                    any => {
                        log::trace!("WaveReader received cmd_rx={:?}", any);
                    }
                }
            }
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
//     fn run(&self, rx: Receiver<Vec<S>>, status_tx: SyncSender<Status>) -> Result<(), IOError> {
//         unimplemented!();
//     }
// }

#[derive(Debug, Default)]
pub struct PAReader {
    info: Info,
    dev: usize,
}

impl PAReader {
    pub fn new(dev: usize, frame: Frame, rate: Hz, ch: Ch) -> Self {
        PAReader {
            info: Info {
                input_frame: 0,
                input_rate: 0,
                input_ch: 0,

                output_frame: frame,
                output_rate: rate,
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
    fn info(&self) -> &Info {
        &self.info
    }
    fn run(
        &mut self,
        tx: SyncSender<Vec<S>>,
        status_tx: SyncSender<Status>,
        cmd_rx: Receiver<Cmd>,
    ) -> Result<()> {
        let pa = pa::PortAudio::new().with_context(|| "could not initialize portaudio")?;
        let ch = self.info.output_ch as i32;
        let interleaved = true;
        let frame = self.info.output_frame as u32;
        let player_info = get_device_info(&pa, self.dev)?;
        let rate = self.info.output_rate as f64;
        let latency = player_info.default_low_input_latency;
        let stream_params = pa::StreamParameters::<S>::new(
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
        // wait Cmd::Start for synchronization (currently no check)
        cmd_rx.recv().unwrap();

        loop {
            // poll commands
            let cmd = cmd_rx.try_recv();
            if cmd.is_ok() {
                match cmd.unwrap() {
                    Cmd::Stop => {
                        log::debug!("PAReader received Cmd::Stop so stop now");
                        break;
                    }
                    any => {
                        log::trace!("PAReader received cmd_rx={:?}", any);
                    }
                }
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
    pub fn new(dev: usize, frame: Frame, rate: Hz, ch: Ch) -> Self {
        PAWriter {
            info: Info {
                input_frame: frame,
                input_rate: rate,
                input_ch: ch,

                output_frame: 0,
                output_rate: 0,
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
    fn info(&self) -> &Info {
        &self.info
    }
    fn run(
        &mut self,
        rx: Receiver<Vec<S>>,
        status_tx: SyncSender<Status>,
        cmd_rx: Receiver<Cmd>,
    ) -> Result<()> {
        let pa = pa::PortAudio::new().with_context(|| "could not initialize portaudio")?;
        let ch = self.info.input_ch as i32;
        let interleaved = true;
        let frame = self.info.input_frame as u32;
        let player_info = get_device_info(&pa, self.dev)?;
        let rate = self.info.input_rate as f64;
        let latency = player_info.default_low_output_latency;
        let stream_params = pa::StreamParameters::<S>::new(
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
        // wait Cmd::Start for synchronization (currently no check)
        cmd_rx.recv().unwrap();

        for buf in rx {
            // poll commands
            let cmd = cmd_rx.try_recv();
            if cmd.is_ok() {
                match cmd.unwrap() {
                    // Upstream closes tx channel so this is not necessary
                    // Cmd::Stop => {
                    //     log::debug!("PAWriter received Cmd::Stop so stop now");
                    //     break;
                    // }
                    any => {
                        log::trace!("PAWriter received cmd_rx={:?}", any);
                    }
                }
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
    pub fn new(cmd: &str, args: &str, frame: Frame, rate: Hz, ch: Ch) -> Self {
        PipeReader {
            info: Info {
                input_frame: 0,
                input_rate: 0,
                input_ch: 0,

                output_frame: frame,
                output_rate: rate,
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
    fn info(&self) -> &Info {
        &self.info
    }
    fn run(
        &mut self,
        tx: SyncSender<Vec<S>>,
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
        log::debug!(
            "PipeReader::new pid={} cmd={} args={}",
            process.id(),
            self.cmd,
            self.args
        );

        status_tx.send(Status::TxInit(IO::Input)).unwrap();
        // wait Cmd::Start for synchronization (currently no check)
        cmd_rx.recv().unwrap();

        let sample_size = std::mem::size_of::<i16>(); // 2
        let buflen = self.info.output_frame * self.info.output_ch * sample_size;
        let mut buf = vec![0; buflen];
        let mut o = process.stdout.take().unwrap();
        loop {
            // poll commands
            let cmd = cmd_rx.try_recv();
            if cmd.is_ok() {
                match cmd.unwrap() {
                    Cmd::Stop => {
                        log::debug!("PipeReader received Cmd::Stop so stop now");
                        // TODO: send SIGTERM to terminate the process gracefully
                        if let Err(e) = process.kill() {
                            log::error!("PipeReader could not kill the process as {}", e);
                        }
                        break;
                    }
                    any => {
                        log::trace!("PipeReader received cmd_rx={:?}", any);
                    }
                }
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
                        // if n % self.info.output_ch * sample_size != 0 {
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

fn apply_filters(l: &[S], r: &[S], lfs: &mut VecFilters, rfs: &mut VecFilters) -> (Vec<S>, Vec<S>) {
    let l = lfs.iter_mut().fold(l.to_vec(), |x, f| f.apply(&x));
    let r = rfs.iter_mut().fold(r.to_vec(), |x, f| f.apply(&x));
    (l, r)
}

fn apply_filters2(l: &[S], r: &[S], sfs: &mut VecFilters2ch) -> (Vec<S>, Vec<S>) {
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
    lr: Resampler,
    rr: Resampler,
    lvf: VecFilters,
    rvf: VecFilters,
    vf2: VecFilters2ch,
}

impl DSP {
    pub fn new(frame: Frame, rate: Hz, vf2_json: &str) -> Self {
        DSP::with_resampling(frame, rate, rate, vf2_json)
    }
    pub fn with_resampling(frame: Frame, from_rate: Hz, to_rate: Hz, vf2_json: &str) -> Self {
        let vf2 = parse_vec2ch(vf2_json, to_rate);
        let (lr, output_frame) = Resampler::new(frame, from_rate, to_rate);
        let (rr, _) = Resampler::new(frame, from_rate, to_rate);
        let p = DSP {
            info: Info {
                input_frame: frame,
                input_rate: from_rate,
                input_ch: 2,

                output_frame,
                output_rate: to_rate,
                output_ch: 2,
            },
            lr,
            rr,
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
    fn info(&self) -> &Info {
        &self.info
    }
    fn run(
        &mut self,
        rx: Receiver<Vec<S>>,
        tx: SyncSender<Vec<S>>,
        status_tx: SyncSender<Status>,
        cmd_rx: Receiver<Cmd>,
    ) -> Result<()> {
        // RMS window = 100 ms
        // peak hold timer = 2000 ms
        let mut l_in_level = LevelMeter1ch::new(
            status_tx.clone(),
            IO::Input,
            0,
            (self.info.input_rate as f32 * 0.1) as usize,
            self.info.input_rate * 2,
        );
        let mut r_in_level = LevelMeter1ch::new(
            status_tx.clone(),
            IO::Input,
            1,
            (self.info.input_rate as f32 * 0.1) as usize,
            self.info.input_rate * 2,
        );
        let mut l_out_level = LevelMeter1ch::new(
            status_tx.clone(),
            IO::Output,
            0,
            (self.info.output_rate as f32 * 0.1) as usize,
            self.info.output_rate * 2,
        );
        let mut r_out_level = LevelMeter1ch::new(
            status_tx.clone(),
            IO::Output,
            1,
            (self.info.output_rate as f32 * 0.1) as usize,
            self.info.output_rate * 2,
        );

        status_tx.send(Status::RxInit(IO::Processor)).unwrap();
        status_tx.send(Status::TxInit(IO::Processor)).unwrap();
        // wait Cmd::Start for synchronization (currently no check)
        cmd_rx.recv().unwrap();

        // send initial filters
        status_tx
            .send(Status::Loaded(vec2ch_to_json(&self.vf2)))
            .unwrap();

        for buf in rx {
            // poll commands
            let cmd = cmd_rx.try_recv();
            if cmd.is_ok() {
                match cmd.unwrap() {
                    Cmd::Reload(s) => {
                        log::debug!("DSP received Cmd::Reload so reload config={}", s);
                        self.vf2 = parse_vec2ch(&s, self.info.output_rate);
                        status_tx
                            .send(Status::Loaded(vec2ch_to_json(&self.vf2)))
                            .unwrap();
                    }
                    // Upstream closes tx channel so this is not necessary
                    // Cmd::Stop => {
                    //     log::debug!("DSP received Cmd::Stop so stop now");
                    //     break;
                    // }
                    any => {
                        log::trace!("DPS received cmd_rx={:?}", any);
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
            let l = self.lr.apply(&l);
            let r = self.rr.apply(&r);
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
    rms_cur: S,
    rms_cnt: u64,
    rms_window: Frame,
    rms_buf: VecDeque<S>,
    peak_cur: S,
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
    pub fn push_data(&mut self, s: &[S]) {
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

fn parse_vec2ch(json: &str, fs: Hz) -> VecFilters2ch {
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

fn setup_vf(fs: Hz) -> (VecFilters, VecFilters) {
    let lvf: VecFilters = vec![
        // BiquadFilter::newb(BQFType::HighPass, fs, 250, 0.0, BQFParam::Q(0.707)),
        // BiquadFilter::newb(BQFType::LowPass, fs, 8000, 0.0, BQFParam::Q(0.707)),
        // BiquadFilter::newb(BQFType::PeakingEQ, fs, 880, 9.0, BQFParam::BW(1.0)),
        // Volume::newb(VolumeCurve::Gain, -6.0),
        // Delay::newb(200, fs),
        NopFilter::newb(),
    ];
    let rvf: VecFilters = vec![
        // BiquadFilter::newb(BQFType::HighPass, fs, 250, 0.0, BQFParam::Q(0.707)),
        // BiquadFilter::newb(BQFType::LowPass, fs, 8000, 0.0, BQFParam::Q(0.707)),
        // BiquadFilter::newb(BQFType::PeakingEQ, fs, 880, 9.0, BQFParam::BW(1.0)),
        // Volume::newb(VolumeCurve::Gain, -6.0),
        // Delay::newb(200, fs),
        NopFilter::newb(),
    ];
    log::info!("filters (L ch): {}", vec_to_json(&lvf));
    log::info!("filters (R ch): {}", vec_to_json(&rvf));
    (lvf, rvf)
}

fn setup_vf2(fs: Hz) -> VecFilters2ch {
    let pf1 = PairedFilter::newb(
        Volume::newb(VolumeCurve::Gain, -12.0),
        Volume::newb(VolumeCurve::Gain, -12.0),
        fs,
    );
    // let pf2 = PairedFilter::newb(ReverbBeta::newb(fs), ReverbBeta::newb(fs), fs);
    let pf3 = CrossfeedBeta::newb(3600, fs);
    let vf2: VecFilters2ch = vec![
        // VocalRemover::newb(VocalRemoverType::RemoveCenter, fs),
        // VocalRemover::newb(VocalRemoverType::RemoveCenterBW(usize::MIN, usize::MAX), fs),
        // VocalRemover::newb(VocalRemoverType::RemoveCenterBW(240, 4400), fs),
        pf3, pf1,
    ];
    log::info!("filters (L&R ch): {}", vec2ch_to_json(&vf2));
    vf2
}

fn gcd(a: Hz, b: Hz) -> Hz {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

#[derive(Debug, Default)]
struct Resampler {
    from_frame: Frame,
    to_frame: Frame,
    from_rate: Hz,
    to_rate: Hz,
    up: usize,
    down: usize,
    lpf: BiquadFilter,
}

impl Resampler {
    pub fn new(frame: Frame, from: Hz, to: Hz) -> (Self, Frame) {
        // e.g., frame=1470, from=44100, to=48000
        //   GCD(44100, 48000) -> 300
        //   Upsampling 160X
        //   Downsampling 147X
        //   LPF f0 = 22050Hz
        //   Output frame = 1470/147*160 = 1600
        let up = to / gcd(from, to);
        let down = from / gcd(from, to);
        let lpf_hz = std::cmp::min(from, to) / 2;
        let to_frame = (frame as f32 / down as f32 * up as f32).round() as Frame;
        let lpf = BiquadFilter::new(BQFType::LowPass, from * up, lpf_hz, 0., BQFParam::Q(0.707));
        let r = Resampler {
            from_frame: frame,
            to_frame,
            from_rate: from,
            to_rate: to,
            up,
            down,
            lpf,
        };
        log::debug!("io::Resampler {:?}", r);
        (r, to_frame)
    }
    pub fn apply(&mut self, x: &[S]) -> Vec<S> {
        if self.up == 1 && self.down == 1 {
            return x.to_vec();
        }
        let mut upsampled: Vec<S>;
        if self.up == 1 {
            upsampled = x.to_vec();
            upsampled = self.lpf.apply(&upsampled);
        } else {
            upsampled = vec![0.0 as S; x.len() * self.up];
            for (i, v) in upsampled.iter_mut().enumerate() {
                // TODO: do interpolation
                *v = x[i / self.up];
            }
            upsampled = self.lpf.apply(&upsampled);
        }
        if self.down == 1 {
            upsampled
        } else {
            upsampled
                .into_iter()
                .enumerate()
                .filter(|&(i, _)| i % self.down == 0)
                .map(|(_, v)| v)
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resampler_new() {
        let t_from = 44100;
        let t_to = 48000;
        let t_frame = t_from / 30;
        let t = DSP::with_resampling(t_frame, t_from, t_to, "[]");
        let want = Info {
            input_frame: 1470,
            input_rate: 44100,
            input_ch: 2,
            output_frame: 1600,
            output_rate: 48000,
            output_ch: 2,
        };
        assert_eq!(t.info(), want);
    }
}
