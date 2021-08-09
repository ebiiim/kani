use crate::err::DeqError;
use crate::pautil;
use deq_io as io;
use portaudio as pa;
use std::io::stdin;
use std::io::{stdout, Write};
use std::process;
use std::sync::mpsc::sync_channel;
use std::thread;

fn print_devices(pa: &pa::PortAudio) -> Result<(), DeqError> {
    let device_names = pautil::get_device_names(pa)?;
    device_names.iter().for_each(|(idx, name)| {
        println!("{}\t{}", idx, name);
    });
    Ok(())
}

fn print_device_info(pa: &pa::PortAudio, idx: usize) -> Result<(), DeqError> {
    let devinfo = pautil::get_device_info(pa, idx)?;
    println!("{:#?}", devinfo);
    Ok(())
}

fn get_default_device(pa: &pa::PortAudio) -> Result<usize, DeqError> {
    let mut dev = usize::MAX;
    let device_names = pautil::get_device_names(pa)?;
    device_names.iter().for_each(|(idx, name)| {
        if name == "default" {
            dev = *idx;
        }
    });
    if dev == usize::MAX {
        Err(DeqError::Device)
    } else {
        Ok(dev)
    }
}

fn read_str(s: &str) -> Result<String, DeqError> {
    print!("{}", s);
    if stdout().flush().is_err() {
        log::error!("could not flush stdout");
    }
    let mut input = String::new();
    if stdin().read_line(&mut input).is_err() {
        log::error!("could not read line");
        return Err(DeqError::Operation);
    }
    log::trace!("read_str={}", input);
    Ok(input.trim().to_string())
}

fn read_int(s: &str) -> Result<usize, DeqError> {
    let read = read_str(s)?.trim().parse();
    log::trace!("read_int={:?}", read);
    match read {
        Ok(i) => Ok(i),
        Err(_) => Err(DeqError::Operation),
    }
}

pub fn play(
    r: Box<dyn io::Input + Send>,
    w: Box<dyn io::Output + Send>,
    dsp: Box<dyn io::Processor + Send>,
) {
    let (tx1, rx1) = sync_channel(0);
    let (tx2, rx2) = sync_channel(0);
    let (in_status_tx, status_rx) = sync_channel(0);
    let dsp_status_tx = in_status_tx.clone();
    let out_status_tx = in_status_tx.clone();

    let frame = r.info().frame;
    let rate = r.info().rate;

    let _ = thread::spawn(move || {
        r.run(tx1, in_status_tx).unwrap();
    });
    let _ = thread::spawn(move || {
        dsp.run(rx1, tx2, dsp_status_tx).unwrap();
    });
    let _ = thread::spawn(move || {
        w.run(rx2, out_status_tx).unwrap();
    });

    // prepare avg latency
    let mut latency_avg = 0.0f64;
    let avg_sec = 3;
    let n = rate as u64 / frame as u64 * avg_sec;
    let mut count: u64 = 0;
    for s in status_rx {
        match s {
            io::Status::Latency(l) => {
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
            _ => {
                log::trace!("{:?}", s)
            }
        }
    }
}

pub fn start(pa: &pa::PortAudio) {
    let term_clear = || print!("\x1B[2J");
    let term_left_top = || print!("\x1B[1;1H");
    let no_dev = 1 << 31;

    let mut input_dev = no_dev;
    let mut output_dev = no_dev;

    term_clear();
    term_left_top();

    loop {
        let cmd = read_int("[1]all devices [2]show details\n[3]input device [4]output device [5]use defaults\n[8]play wav [9]play\n[0]quit\n>");
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
                        if pautil::get_device_info(pa, n).is_ok() {
                            input_dev = n;
                        } else {
                            log::error!("could not find device id={}", n);
                        }
                    }
                } else if cmd == 4 {
                    if let Ok(n) = read_int("output device> ") {
                        if pautil::get_device_info(pa, n).is_ok() {
                            output_dev = n;
                        } else {
                            log::error!("could not find device id={}", n);
                        }
                    }
                } else if cmd == 5 {
                    if let Ok(dev) = get_default_device(pa) {
                        input_dev = dev;
                        output_dev = dev;
                    } else {
                        log::error!("could not find default device");
                    }
                } else if cmd == 8 {
                    if output_dev == no_dev {
                        log::error!("please select output device first");
                        continue;
                    }
                    if let Ok(n) = read_str("file name> ") {
                        let frame = 1024;
                        let r = io::WaveReader::new(frame, &n).unwrap();
                        let dsp = io::DSP::new(r.info().frame, r.info().rate);
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
                    if input_dev == no_dev || output_dev == no_dev {
                        log::error!("please select devices first");
                        continue;
                    }
                    if let Err(e) = pautil::play(pa, input_dev, output_dev) {
                        log::error!("play err={}", e);
                        process::exit(0);
                    }
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
