use portaudio as pa;
use std::io::stdin;
use std::io::{stdout, Write};
use std::process;

use crate::err::DeqError;
use crate::pautil;

fn print_devices(pa: &pa::PortAudio) -> Result<(), DeqError> {
    let device_names = pautil::get_device_names(pa)?;
    device_names.iter().all(|(idx, name)| {
        println!("{}\t{}", idx, name);
        true
    });
    Ok(())
}

fn print_device_info(pa: &pa::PortAudio, idx: usize) -> Result<(), DeqError> {
    let devinfo = pautil::get_device_info(pa, idx)?;
    println!("{:#?}", devinfo);
    Ok(())
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

pub fn start(pa: &pa::PortAudio) {
    let term_clear = || print!("\x1B[2J");
    let term_left_top = || print!("\x1B[1;1H");
    let no_dev = 1 << 31;

    let mut input_dev = no_dev;
    let mut output_dev = no_dev;

    term_clear();
    term_left_top();

    loop {
        let cmd = read_int("[1]devices [2]detail [3]input [4]output [8]play wav [9]play [0]quit >");
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
                } else if cmd == 8 {
                    if output_dev == no_dev {
                        log::error!("please select output device first");
                        continue;
                    }
                    if let Ok(n) = read_str("file name> ") {
                        if let Err(e) = pautil::play_wav(pa, &n, output_dev) {
                            log::error!("play wav err={}", e);
                            process::exit(0);
                        }
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
