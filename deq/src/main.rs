use anyhow::{bail, Context, Result};
use deq_io as io;
use io::Status::*;
use portaudio as pa;
use std::env;
use std::io::{stdin, stdout, Write};
use std::process;
use std::sync::mpsc::sync_channel;
use std::thread;

extern crate deq_filter;
extern crate deq_io;

const PATH_FILTERS: &str = "filters.json";

#[cfg(any(target_os = "linux", target_os = "macos"))]
const PATH_LIBRESPOT: &str = "librespot";
#[cfg(target_os = "windows")]
const PATH_LIBRESPOT: &str = "librespot.exe";

#[cfg(target_os = "linux")]
const DEFAULT_INPUT_DEV: &str = "default";
#[cfg(target_os = "linux")]
const DEFAULT_OUTPUT_DEV: &str = "default";
#[cfg(target_os = "windows")]
const DEFAULT_INPUT_DEV: &str = "Microsoft Sound Mapper - Input";
#[cfg(target_os = "windows")]
const DEFAULT_OUTPUT_DEV: &str = "Microsoft Sound Mapper - Output";

const FRAME: usize = 1024;
const CLI_DEV_INVALID: usize = usize::MAX;
const TERM_CLEAR: &str = "\x1B[2J";
const TERM_LEFT_TOP: &str = "\x1B[1;1H";

fn main() {
    // init logger
    let log_levels = ["error", "warn", "info", "debug", "trace"];
    let log_default = log_levels[2]; // default log level: info
    let log_env = "RUST_LOG";
    match env::var(log_env) {
        Err(_) => env::set_var(log_env, log_default),
        Ok(l) => {
            if log_levels.iter().filter(|s| s.to_string() == l).count() != 1 {
                env::set_var(log_env, log_default);
            }
        }
    };
    env_logger::init();

    // start CLI
    start();
}

fn print_devices(pa: &pa::PortAudio) -> Result<()> {
    let device_names = io::get_device_names(pa)?;
    device_names.iter().for_each(|(idx, name)| {
        println!("{}\t{}", idx, name);
    });
    Ok(())
}

fn print_device_info(pa: &pa::PortAudio, idx: usize) -> Result<()> {
    let devinfo = io::get_device_info(pa, idx)?;
    println!("{:#?}", devinfo);
    Ok(())
}

fn get_device_by_name(pa: &pa::PortAudio, n: &str) -> Result<usize> {
    let mut dev = CLI_DEV_INVALID;
    let device_names = io::get_device_names(pa)?;
    device_names.iter().for_each(|(idx, name)| {
        if name == n {
            dev = *idx;
        }
    });
    if dev == CLI_DEV_INVALID {
        bail!("device \"{}\" not found", n);
    } else {
        Ok(dev)
    }
}

fn get_default_devices(pa: &pa::PortAudio) -> (usize, usize) {
    let (mut input_dev, mut output_dev) = (CLI_DEV_INVALID, CLI_DEV_INVALID);
    if let Ok(dev) = get_device_by_name(pa, DEFAULT_INPUT_DEV) {
        input_dev = dev;
    } else {
        log::error!("could not find device \"{}\"", DEFAULT_INPUT_DEV);
    }
    if let Ok(dev) = get_device_by_name(pa, DEFAULT_OUTPUT_DEV) {
        output_dev = dev;
    } else {
        log::error!("could not find device \"{}\"", DEFAULT_OUTPUT_DEV);
    }
    (input_dev, output_dev)
}

fn read_str(s: &str) -> Result<String> {
    print!("{}", s);
    stdout().flush().with_context(|| "coult not flush stdout")?;
    let mut input = String::new();
    stdin()
        .read_line(&mut input)
        .with_context(|| "coult not read line")?;
    log::trace!("read_str={}", input);
    Ok(input.trim().to_string())
}

fn read_int(s: &str) -> Result<usize> {
    let read = read_str(s)?.trim().parse();
    if read.is_err() {
        bail!("could not parse {} to int", s);
    }
    let read = read.unwrap();
    log::trace!("read_int={:?}", read);
    Ok(read)
}

pub fn play(
    mut r: Box<dyn io::Input + Send>,
    mut w: Box<dyn io::Output + Send>,
    mut dsp: Box<dyn io::Processor + Send>,
) {
    let (tx1, rx1) = sync_channel(0);
    let (tx2, rx2) = sync_channel(0);
    let (in_status_tx, status_rx) = sync_channel(0);
    let dsp_status_tx = in_status_tx.clone();
    let out_status_tx = in_status_tx.clone();
    let (in_cmd_tx, in_cmd_rx) = sync_channel(0);
    let (dsp_cmd_tx, dsp_cmd_rx) = sync_channel(0);
    let (out_cmd_tx, out_cmd_rx) = sync_channel(0);

    let frame = r.info().frame;
    let rate = r.info().rate;

    let _ = thread::spawn(move || {
        r.run(tx1, in_status_tx, in_cmd_rx).unwrap();
    });
    let _ = thread::spawn(move || {
        dsp.run(rx1, tx2, dsp_status_tx, dsp_cmd_rx).unwrap();
    });
    let _ = thread::spawn(move || {
        w.run(rx2, out_status_tx, out_cmd_rx).unwrap();
    });

    // prepare avg latency
    let mut latency_avg = 0.0f64;
    let avg_sec = 10;
    let n = rate as u64 / frame as u64 * avg_sec;
    let mut count: u64 = 0;
    for s in status_rx {
        match s {
            // check io::Status::*
            Latency(l) => {
                latency_avg -= latency_avg / n as f64;
                latency_avg += l as f64 / n as f64;
                count += 1;
                if count % n == 0 {
                    log::info!(
                        "filter latency ({}s avg): {:.3} ms | total frames: {}",
                        avg_sec,
                        latency_avg / 1000.0,
                        count
                    );
                }
            }
            Interpolated(_) => {
                log::info!("{:?}", s);
            }
            TxInit(_) | RxInit(_) => {
                log::info!("{:?}", s)
            }
            TxErr(_) | RxErr(_) => {
                log::warn!("{:?}", s)
            }
            _ => {
                log::trace!("{:?}", s)
            }
        }
    }
}

pub fn start() {
    let pa = pa::PortAudio::new();
    if pa.is_err() {
        log::error!("could not initialize portaudio: {:?}", pa.unwrap_err());
        process::exit(0);
    }
    let pa = &pa.unwrap();

    let mut input_dev = CLI_DEV_INVALID;
    let mut output_dev = CLI_DEV_INVALID;
    print!("{}", TERM_CLEAR);
    print!("{}", TERM_LEFT_TOP);
    loop {
        let cmd = read_int(
            "[1]list all devices [2]show device info
[3]select input device [4]select output device [5]use defaults (Linux|Windows)
[7]play {Spotify->DSP->Output} (requires: librespot binary in current dir)
[8]play {WAVE->DSP->Output}
[9]play {Input->DSP->Output}
[0]quit
>",
        );
        match cmd {
            Err(_) => {
                continue;
            }
            Ok(cmd) => {
                if cmd == 1 {
                    println!("PortAudio version: {}", pa::version());
                    let dc = pa.device_count();
                    if dc.is_err() {
                        log::error!("could not fetch devices");
                        continue;
                    }
                    println!("Device count: {}", dc.unwrap());
                    if print_devices(pa).is_err() {
                        log::error!("could not print devices");
                    }
                } else if cmd == 2 {
                    if let Ok(n) = read_int("see device> ") {
                        if print_device_info(pa, n).is_err() {
                            log::error!("could not find device id={}", n);
                        }
                    }
                } else if cmd == 3 {
                    if let Ok(n) = read_int("input device> ") {
                        if io::get_device_info(pa, n).is_ok() {
                            input_dev = n;
                        } else {
                            log::error!("could not find device id={}", n);
                        }
                    }
                    log::debug!("input_dev={}, output_dev={}", input_dev, output_dev);
                } else if cmd == 4 {
                    if let Ok(n) = read_int("output device> ") {
                        if io::get_device_info(pa, n).is_ok() {
                            output_dev = n;
                        } else {
                            log::error!("could not find device id={}", n);
                        }
                    }
                    log::debug!("input_dev={}, output_dev={}", input_dev, output_dev);
                } else if cmd == 5 {
                    let (indev, outdev) = get_default_devices(pa);
                    input_dev = indev;
                    output_dev = outdev;
                    log::debug!("input_dev={}, output_dev={}", input_dev, output_dev);
                } else if cmd == 7 {
                    if output_dev == CLI_DEV_INVALID {
                        log::error!("please select output device first");
                        continue;
                    }
                    let r = io::PipeReader::new(
                        PATH_LIBRESPOT,
                        "-n DEQ(Librespot) -b 320 --backend pipe",
                        FRAME,
                        44100,
                        2,
                    );
                    let dsp = io::DSP::new(
                        r.info().frame,
                        r.info().rate,
                        &std::fs::read_to_string(PATH_FILTERS).unwrap_or(String::from("")),
                    );
                    let w = io::PAWriter::new(
                        output_dev,
                        r.info().frame,
                        r.info().rate,
                        r.info().output_ch,
                    );
                    let r = Box::new(r) as Box<dyn io::Input + Send>;
                    let w = Box::new(w) as Box<dyn io::Output + Send>;
                    let dsp = Box::new(dsp) as Box<dyn io::Processor + Send>;
                    play(r, w, dsp);
                } else if cmd == 8 {
                    if output_dev == CLI_DEV_INVALID {
                        log::error!("please select output device first");
                        continue;
                    }
                    if let Ok(n) = read_str("file name> ") {
                        let frame = FRAME;
                        let r = io::WaveReader::new(frame, &n).unwrap();
                        let dsp = io::DSP::new(
                            r.info().frame,
                            r.info().rate,
                            &std::fs::read_to_string(PATH_FILTERS).unwrap_or(String::from("")),
                        );
                        let w = io::PAWriter::new(
                            output_dev,
                            r.info().frame,
                            r.info().rate,
                            r.info().output_ch,
                        );
                        let r = Box::new(r) as Box<dyn io::Input + Send>;
                        let w = Box::new(w) as Box<dyn io::Output + Send>;
                        let dsp = Box::new(dsp) as Box<dyn io::Processor + Send>;
                        play(r, w, dsp);
                    }
                } else if cmd == 9 {
                    if input_dev == CLI_DEV_INVALID || output_dev == CLI_DEV_INVALID {
                        log::error!("please select devices first");
                        continue;
                    }
                    let frame = FRAME;
                    let r = io::PAReader::new(input_dev, frame, 48000, 2);
                    let dsp = io::DSP::new(
                        r.info().frame,
                        r.info().rate,
                        &std::fs::read_to_string(PATH_FILTERS).unwrap_or(String::from("")),
                    );
                    let w = io::PAWriter::new(
                        output_dev,
                        r.info().frame,
                        r.info().rate,
                        r.info().output_ch,
                    );
                    let r = Box::new(r) as Box<dyn io::Input + Send>;
                    let w = Box::new(w) as Box<dyn io::Output + Send>;
                    let dsp = Box::new(dsp) as Box<dyn io::Processor + Send>;
                    play(r, w, dsp);
                } else if cmd == 0 {
                    log::debug!("exit");
                    process::exit(0);
                } else {
                    log::debug!("invalid command");
                    continue;
                }
            }
        }
    }
}
