use anyhow::{bail, Context, Result};
use getopts::Options;
use io::Status::*;
use kani_io as io;
use portaudio as pa;
use std::env;
use std::io::{stdin, stdout, Write};
use std::process;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::sync_channel;
use std::sync::Arc;
use std::thread;

extern crate kani_filter;
extern crate kani_io;

const PATH_FILTERS: &str = "filters.json";

#[cfg(any(target_os = "linux", target_os = "macos"))]
const PATH_LIBRESPOT: &str = "./librespot";
#[cfg(target_os = "windows")]
const PATH_LIBRESPOT: &str = "librespot.exe";

#[cfg(target_os = "linux")]
const DEFAULT_INPUT_DEV: &str = "default";
#[cfg(target_os = "linux")]
const DEFAULT_OUTPUT_DEV: &str = "default";
#[cfg(target_os = "macos")]
const DEFAULT_INPUT_DEV: &str = "undefined";
#[cfg(target_os = "macos")]
const DEFAULT_OUTPUT_DEV: &str = "undefined";
#[cfg(target_os = "windows")]
const DEFAULT_INPUT_DEV: &str = "Microsoft Sound Mapper - Input";
#[cfg(target_os = "windows")]
const DEFAULT_OUTPUT_DEV: &str = "Microsoft Sound Mapper - Output";

const DEFAULT_FRAME: usize = 1024;
const CLI_DEV_INVALID: usize = usize::MAX;
const TERM_CLEAR: &str = "\x1B[2J";
const TERM_LEFT_TOP: &str = "\x1B[1;1H";
const TERM_C_END: &str = "\x1B[0m";
const TERM_C_LRED: &str = "\x1B[91m";
const TERM_C_LGREEN: &str = "\x1B[92m";
const TERM_C_LYELLOW: &str = "\x1B[93m";
const TERM_C_LBLUE: &str = "\x1B[93m";
const TERM_C_LMAGENTA: &str = "\x1B[95m";
const TERM_C_LCYAN: &str = "\x1B[96m";

fn main() {
    // init logger
    let log_levels = ["error", "warn", "info", "debug", "trace"];
    let log_default = log_levels[1]; // default log level: warn
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

    // args
    let args: Vec<String> = env::args().collect();
    let bin = args[0].clone();
    let mut opts = Options::new();
    opts.optopt(
        "f",
        "",
        &format!("set frame size (default: {})", DEFAULT_FRAME),
        "FRAME",
    );
    opts.optflag("h", "help", "print usage");
    opts.optflag("n", "no-level-meter", "no show level meter");
    let m = match opts.parse(&args[1..]) {
        Ok(m) => m,
        Err(e) => {
            panic!("{}", e)
        }
    };
    if m.opt_present("h") {
        print_usage(&bin, opts);
        process::exit(0);
    }
    let frame = m
        .opt_str("f")
        .unwrap_or(String::from(""))
        .parse::<usize>()
        .unwrap_or(DEFAULT_FRAME);
    let no_level_meter = m.opt_present("n");
    start(frame, no_level_meter);
}

fn print_usage(bin: &str, opts: Options) {
    let usage = format!("Usage: {} [-f FRAME] [-n]", bin);
    print!("{}", opts.usage(&usage));
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
    no_level_meter: bool,
) {
    // use sync channel to pace the reader so do not use async channel
    // use small buffer and let channels no rendezvous
    let (tx1, rx1) = sync_channel(1);
    let (tx2, rx2) = sync_channel(1);

    // please receive!
    let (in_status_tx, status_rx) = sync_channel(0);
    let dsp_status_tx = in_status_tx.clone();
    let out_status_tx = in_status_tx.clone();

    // receivers do try_recv() so use try_send and let buffer>0
    let (in_cmd_tx, in_cmd_rx) = sync_channel(1);
    let (dsp_cmd_tx, dsp_cmd_rx) = sync_channel(1);
    let (out_cmd_tx, out_cmd_rx) = sync_channel(1);

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

    let mut current_vf2 = read_vf2();

    // CLI
    // do increment in Latency as Latency is received once per frame
    let mut count: u64 = 0;
    // prepare avg latency
    let mut latency_avg = 0.0f64;
    let avg_sec = 2;
    let n = rate as u64 / frame as u64 * avg_sec;
    // prepare RMS and peak
    let mut l_rms_in = 0.0;
    let mut l_rms_out = 0.0;
    let mut l_peak_in = 0.0;
    let mut l_peak_out = 0.0;
    let mut r_rms_in = 0.0;
    let mut r_rms_out = 0.0;
    let mut r_peak_in = 0.0;
    let mut r_peak_out = 0.0;
    let calc_gain = |v: f32| 20.0 * v.log10();
    let draw_bar = |cur: isize, peak: isize, max: isize, step: isize| {
        let yellow = 3;
        let mut s = String::from("");
        for i in 0..((max + 2) / step) {
            if i == (peak / step) && i > max / step - yellow {
                s += &format!("{}|{}", TERM_C_LYELLOW, TERM_C_END);
            } else if i == (peak / step) {
                s += &format!("{}|{}", TERM_C_LGREEN, TERM_C_END);
            } else if i < (cur / step) && i > max / step - yellow {
                s += &format!("{}|{}", TERM_C_LYELLOW, TERM_C_END);
            } else if i < (cur / step) {
                s += &format!("{}|{}", TERM_C_LGREEN, TERM_C_END);
            } else if i > max / step - yellow {
                s += &format!("{}.{}", TERM_C_LYELLOW, TERM_C_END);
            } else {
                s += &format!("{}.{}", TERM_C_LGREEN, TERM_C_END);
            }
        }
        s
    };
    let draw_volume = |ch: &str, rms: f32, peak: f32, max: isize, step: isize| {
        let mut over1 = format!("{}.{}", TERM_C_LRED, TERM_C_END);
        let mut over2 = String::from("");
        let rms_db = calc_gain(rms);
        let peak_db = calc_gain(peak);
        // clipping occurs in i16
        if peak_db > 1.0 / 32768.0 {
            over1 = format!("{}|{}", TERM_C_LRED, TERM_C_END);
            over2 = format!(" {}OVR{}", TERM_C_LRED, TERM_C_END);
        }
        // L |||||||||||||||||||||||........| +0.8 OVR
        // R ||||||||||||||||||...........|.. -1.2
        format!(
            "{} {}{} {:+.1}{}\n",
            ch,
            draw_bar(
                max + rms_db.ceil() as isize,
                max + peak_db.ceil() as isize,
                max,
                step
            ),
            over1,
            peak_db,
            over2,
        )
    };
    let draw_time = |mut sec: usize| {
        let h = sec / 3600;
        sec %= 3600;
        let m = sec / 60;
        sec %= 60;
        format!("{:02}:{:02}:{:02}", h, m, sec)
    };

    // press Enter to stop
    let sig = Arc::new(AtomicBool::new(false));
    let sig2 = sig.clone();
    let _ = thread::spawn(move || {
        // TODO: need press Enter once anyway
        read_str("").unwrap(); // wait for input
        sig2.store(true, Ordering::Relaxed);
    });

    for s in status_rx {
        // stop?
        if sig.load(Ordering::Relaxed) {
            sig.store(false, Ordering::Relaxed);
            if in_cmd_tx.try_send(io::Cmd::Stop).is_err()
                || dsp_cmd_tx.try_send(io::Cmd::Stop).is_err()
                || out_cmd_tx.try_send(io::Cmd::Stop).is_err()
            {
                log::error!("could not send stop");
                return;
            }
        }
        match s {
            // check io::Status::*
            Latency(l) => {
                latency_avg -= latency_avg / n as f64;
                latency_avg += l as f64 / n as f64;
                count += 1;
                // once every avg_sec
                if count % n == 0 {
                    log::debug!(
                        "DSP latency ({}s avg): {:.3} ms | total frames: {}",
                        avg_sec,
                        latency_avg / 1000.0,
                        count
                    );
                }
                // once every one avg_sec
                if count % (n / avg_sec) == 0 {
                    let tmp = read_vf2();
                    if &tmp != &current_vf2 {
                        log::info!("reload filters");
                        current_vf2 = tmp;
                        if let Err(e) = dsp_cmd_tx.try_send(io::Cmd::Reload(current_vf2.clone())) {
                            log::warn!("could not reload filters err={}", e)
                        }
                    }
                }
                // draw CLI
                if no_level_meter {
                    continue;
                }
                let fps = 30;
                if count % ((rate / frame) as f32 / fps as f32) as u64 == 0 {
                    let status = format!(
                        "Time\t{}\nLatency\t{:.2} ms\n",
                        draw_time(count as usize / (rate / frame)),
                        latency_avg / 1000.0,
                    );
                    let l_in_bar = draw_volume("L", l_rms_in, l_peak_in, 60, 2);
                    let r_in_bar = draw_volume("R", r_rms_in, r_peak_in, 60, 2);
                    let l_out_bar = draw_volume("L", l_rms_out, l_peak_out, 60, 2);
                    let r_out_bar = draw_volume("R", r_rms_out, r_peak_out, 60, 2);
                    let note = "use Enter/Return to stop\n";
                    print!(
                        "{}{}{}\nINPUT\n{}{}\nOUTPUT\n{}{}\n{}",
                        TERM_CLEAR,
                        TERM_LEFT_TOP,
                        status,
                        l_in_bar,
                        r_in_bar,
                        l_out_bar,
                        r_out_bar,
                        note
                    )
                }
            }
            RMS(pos, ch, v) => {
                if pos == io::IO::Input {
                    if ch == 0 {
                        l_rms_in = v;
                    } else {
                        r_rms_in = v;
                    }
                } else {
                    if ch == 0 {
                        l_rms_out = v;
                    } else {
                        r_rms_out = v;
                    }
                }
                log::trace!(
                    "RMS\tL {:.3}=>{:.3} | R {:.3}=>{:.3}",
                    calc_gain(l_rms_in),
                    calc_gain(l_rms_out),
                    calc_gain(r_rms_in),
                    calc_gain(r_rms_out)
                );
            }
            Peak(pos, ch, v) => {
                if pos == io::IO::Input {
                    if ch == 0 {
                        l_peak_in = v;
                    } else {
                        r_peak_in = v;
                    }
                } else {
                    if ch == 0 {
                        l_peak_out = v;
                    } else {
                        r_peak_out = v;
                    }
                }
                log::trace!(
                    "Peak\tL {:.3}=>{:.3} | R {:.3}=>{:.3}",
                    calc_gain(l_peak_in),
                    calc_gain(l_peak_out),
                    calc_gain(r_peak_in),
                    calc_gain(r_peak_out)
                );
            }
            Interpolated(_) => {
                log::info!("{:?}", s);
            }
            TxInit(_) | RxInit(_) => {
                log::info!("{:?}", s)
            }
            TxFin(_) | RxFin(_) => {
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

fn read_vf2() -> String {
    std::fs::read_to_string(PATH_FILTERS).unwrap_or(String::from(""))
}

pub fn start(frame: usize, no_level_meter: bool) {
    let pa = pa::PortAudio::new();
    if pa.is_err() {
        log::error!("could not initialize portaudio: {:?}", pa.unwrap_err());
        process::exit(0);
    }
    let pa = &pa.unwrap();

    let mut input_dev = CLI_DEV_INVALID;
    let mut output_dev = CLI_DEV_INVALID;
    print!("{}{}", TERM_CLEAR, TERM_LEFT_TOP);
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
                    match pa.device_count() {
                        Ok(dc) => println!("Device count: {}", dc),
                        Err(e) => {
                            log::error!("could not fetch devices as {}", e);
                            continue;
                        }
                    }
                    if let Err(e) = print_devices(pa) {
                        log::error!("could not print devices as {}", e);
                    }
                } else if cmd == 2 {
                    if let Ok(n) = read_int("see device> ") {
                        if let Err(e) = print_device_info(pa, n) {
                            log::error!("could not find device id={} as {}", n, e);
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
                        "-n kani -b 320 --backend pipe --initial-volume 100",
                        frame,
                        44100,
                        2,
                    );
                    let dsp = io::DSP::new(r.info().frame, r.info().rate, &read_vf2());
                    let w = io::PAWriter::new(
                        output_dev,
                        r.info().frame,
                        r.info().rate,
                        r.info().output_ch,
                    );
                    let r = Box::new(r) as Box<dyn io::Input + Send>;
                    let w = Box::new(w) as Box<dyn io::Output + Send>;
                    let dsp = Box::new(dsp) as Box<dyn io::Processor + Send>;
                    play(r, w, dsp, no_level_meter);
                    print!("{}{}", TERM_CLEAR, TERM_LEFT_TOP);
                } else if cmd == 8 {
                    if output_dev == CLI_DEV_INVALID {
                        log::error!("please select output device first");
                        continue;
                    }
                    if let Ok(n) = read_str("file name> ") {
                        let r = io::WaveReader::new(frame, &n).unwrap();
                        let dsp = io::DSP::new(r.info().frame, r.info().rate, &read_vf2());
                        let w = io::PAWriter::new(
                            output_dev,
                            r.info().frame,
                            r.info().rate,
                            r.info().output_ch,
                        );
                        let r = Box::new(r) as Box<dyn io::Input + Send>;
                        let w = Box::new(w) as Box<dyn io::Output + Send>;
                        let dsp = Box::new(dsp) as Box<dyn io::Processor + Send>;
                        play(r, w, dsp, no_level_meter);
                        print!("{}{}", TERM_CLEAR, TERM_LEFT_TOP);
                    }
                } else if cmd == 9 {
                    if input_dev == CLI_DEV_INVALID || output_dev == CLI_DEV_INVALID {
                        log::error!("please select devices first");
                        continue;
                    }
                    let r = io::PAReader::new(input_dev, frame, 48000, 2);
                    let dsp = io::DSP::new(r.info().frame, r.info().rate, &read_vf2());
                    let w = io::PAWriter::new(
                        output_dev,
                        r.info().frame,
                        r.info().rate,
                        r.info().output_ch,
                    );
                    let r = Box::new(r) as Box<dyn io::Input + Send>;
                    let w = Box::new(w) as Box<dyn io::Output + Send>;
                    let dsp = Box::new(dsp) as Box<dyn io::Processor + Send>;
                    play(r, w, dsp, no_level_meter);
                    print!("{}{}", TERM_CLEAR, TERM_LEFT_TOP);
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
