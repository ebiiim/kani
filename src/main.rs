use portaudio;
use std::error;
use std::fmt;
use std::io::stdin;
use std::io::{stdout, Write};
use std::process;

#[derive(PartialEq, Eq, Debug)]
enum EqError {
    InvalidDevice,
    InvalidOperation,
}

impl fmt::Display for EqError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let msg = match *self {
            EqError::InvalidDevice => "invalid device id",
            EqError::InvalidOperation => "invalid user operation",
        };
        f.write_str(msg)
    }
}

impl error::Error for EqError {}

fn get_device_names(pa: &portaudio::PortAudio) -> Result<Vec<(usize, String)>, EqError> {
    let devs = pa.devices();
    match devs {
        Ok(devs) => Ok(devs
            .map(|dev| {
                let d = dev.unwrap();
                let portaudio::DeviceIndex(idx) = d.0;
                (idx as usize, String::from(d.1.name))
            })
            .collect()),
        Err(_) => Err(EqError::InvalidDevice),
    }
}

fn print_devices(pa: &portaudio::PortAudio) -> Result<(), EqError> {
    let device_names = get_device_names(&pa)?;
    device_names.iter().all(|(idx, name)| {
        println!("{}\t{}", idx, name);
        true
    });
    Ok(())
}

fn get_device_info(
    pa: &portaudio::PortAudio,
    idx: usize,
) -> Result<portaudio::DeviceInfo, EqError> {
    let devidx = portaudio::DeviceIndex(idx as u32);
    let devs = pa.devices();
    if let Ok(devs) = devs {
        for d in devs {
            if let Ok(d) = d {
                if d.0 == devidx {
                    return Ok(d.1);
                }
            } else {
                return Err(EqError::InvalidDevice);
            }
        }
    }
    Err(EqError::InvalidDevice)
}

fn print_device_info(pa: &portaudio::PortAudio, idx: usize) -> Result<(), EqError> {
    let devinfo = get_device_info(&pa, idx)?;
    println!("{:#?}", devinfo);
    Ok(())
}

fn read_str(s: &str) -> Result<String, EqError> {
    print!("{}", s);
    stdout().flush().unwrap();
    let mut input = String::new();
    if stdin().read_line(&mut input).is_err() {
        return Err(EqError::InvalidOperation);
    }
    Ok(input)
}

fn read_int(s: &str) -> Result<usize, EqError> {
    match read_str(s)?.trim().parse() {
        Ok(i) => Ok(i),
        Err(_) => Err(EqError::InvalidOperation),
    }
}

fn select_devices_loop(pa: &portaudio::PortAudio) -> (usize, usize) {
    let term_clear = || print!("\x1B[2J");
    let term_left_top = || print!("\x1B[1;1H");
    let no_dev = 1 << 31;

    let mut input_dev = no_dev;
    let mut output_dev = no_dev;

    term_clear();
    term_left_top();

    loop {
        if input_dev != no_dev && output_dev != no_dev {
            break;
        }
        let cmd = read_int("Command(0:devices,1:detail,2:input,3:output,9:exit)>");
        match cmd {
            Err(_) => {
                continue;
            }
            Ok(cmd) => {
                if cmd == 0 {
                    println!("PortAudio version: {}", portaudio::version());
                    let dc = pa.device_count();
                    if dc.is_err() {
                        println!("could not fetch devices");
                        continue;
                    }
                    println!("Device count: {}", dc.unwrap());
                    if print_devices(&pa).is_err() {
                        println!("could not print devices");
                    }
                } else if cmd == 1 {
                    let n = read_int("see device> ");
                    match n {
                        Ok(n) => {
                            let _ = print_device_info(&pa, n).is_ok();
                            ()
                        }
                        Err(_) => {}
                    }
                } else if cmd == 2 {
                    let n = read_int("input device> ");
                    match n {
                        Ok(n) => {
                            if get_device_info(&pa, n).is_ok() {
                                input_dev = n;
                            }
                            ()
                        }
                        Err(_) => {}
                    }
                } else if cmd == 3 {
                    let n = read_int("output device> ");
                    match n {
                        Ok(n) => {
                            if get_device_info(&pa, n).is_ok() {
                                output_dev = n;
                            }
                            ()
                        }
                        Err(_) => {}
                    }
                } else if cmd == 9 {
                    process::exit(0);
                } else {
                    continue;
                }
            }
        }
    }
    return (input_dev, output_dev);
}

fn main() {
    let pa = portaudio::PortAudio::new().unwrap();
    let (input_dev, output_dev) = select_devices_loop(&pa);
    println!("Input device: {}, Output device: {}", input_dev, output_dev);
}
