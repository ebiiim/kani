use anyhow::{bail, Context, Result};
use getopts::Options;
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Request, Response, Server};
use hyper::{Method, StatusCode};
use kani_io as io;
use portaudio as pa;
use serde_json;
use std::convert::Infallible;
use std::env;
use std::io::{stdin, stdout, Write};
use std::net::SocketAddr;
use std::process;
use std::sync::Arc;

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

async fn kani_service(
    req: Request<Body>,
    k: Arc<kani::Kani>,
) -> Result<Response<Body>, Infallible> {
    let mut resp = Response::new(Body::empty());

    match (req.method(), req.uri().path()) {
        (&Method::GET, "/") => {
            *resp.body_mut() = Body::from(format!(
                "{} {}",
                env!("CARGO_BIN_NAME"),
                env!("CARGO_PKG_VERSION")
            ));
        }
        (&Method::GET, "/player_info") => {
            let i = k.info();
            *resp.body_mut() = Body::from(serde_json::to_string(&i).unwrap());
        }
        (&Method::GET, "/current_status") => {
            let s = k.status();
            *resp.body_mut() = Body::from(serde_json::to_string(&s).unwrap());
        }
        (&Method::GET, "/filters") => {
            let f = k.filters();
            *resp.body_mut() = Body::from(f);
        }
        (&Method::POST, "/filters") => {
            let full_body = hyper::body::to_bytes(req.into_body()).await.unwrap();
            k.set_filters(&String::from_utf8(full_body.to_vec()).unwrap())
                .unwrap();
            *resp.body_mut() = Body::empty();
        }
        (&Method::POST, "/ctrl/stop") => {
            k.stop().unwrap();
            *resp.body_mut() = Body::empty();
        }
        _ => {
            *resp.status_mut() = StatusCode::NOT_FOUND;
        }
    };
    Ok(resp)
}

#[tokio::main]
async fn main() {
    init_logger();
    let apiargs = parse_args_or_exit(env::args().collect());
    let k = start_player_or_exit(apiargs.frame);

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    let svc = make_service_fn(move |_conn| {
        let k = k.clone();
        async move { Ok::<_, Infallible>(service_fn(move |req| kani_service(req, k.clone()))) }
    });
    let server = Server::bind(&addr).serve(svc);
    let server = server.with_graceful_shutdown((|| async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            log::error!("could not setup signal handler as {}", e);
        }
    })());

    if let Err(e) = server.await {
        log::error!("server error err={}", e);
        std::process::exit(1);
    }
    log::info!("bye");
}

fn init_logger() {
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
}

struct ApiArgs {
    frame: usize,
}

fn print_usage(bin: &str, opts: Options) {
    let usage = format!("Usage: {} [-f FRAME]", bin);
    print!("{}", opts.usage(&usage));
}

fn parse_args_or_exit(args: Vec<String>) -> ApiArgs {
    let bin = args[0].clone();
    let mut opts = Options::new();
    opts.optopt(
        "f",
        "",
        &format!("set frame size (default: {})", DEFAULT_FRAME),
        "FRAME",
    );
    opts.optflag("h", "help", "print usage");
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
    ApiArgs { frame }
}

fn start_player_or_exit(frame: usize) -> Arc<kani::Kani> {
    let pa = pa::PortAudio::new();
    if pa.is_err() {
        log::error!("could not initialize portaudio: {:?}", pa.unwrap_err());
        process::exit(0);
    }
    let pa = &pa.unwrap();

    let mut input_dev = CLI_DEV_INVALID;
    let mut output_dev = CLI_DEV_INVALID;
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
                    let vf2 = std::fs::read_to_string(PATH_FILTERS).unwrap_or(String::from(""));
                    let dsp = io::DSP::new(r.info().frame, r.info().rate, &vf2);
                    let w = io::PAWriter::new(
                        output_dev,
                        r.info().frame,
                        r.info().rate,
                        r.info().output_ch,
                    );
                    let r = Box::new(r) as Box<dyn io::Input + Send>;
                    let w = Box::new(w) as Box<dyn io::Output + Send>;
                    let dsp = Box::new(dsp) as Box<dyn io::Processor + Send>;
                    return kani::Kani::start(r, w, dsp).unwrap();
                } else if cmd == 8 {
                    if output_dev == CLI_DEV_INVALID {
                        log::error!("please select output device first");
                        continue;
                    }
                    if let Ok(n) = read_str("file name> ") {
                        let r = io::WaveReader::new(frame, &n).unwrap();
                        let vf2 = std::fs::read_to_string(PATH_FILTERS).unwrap_or(String::from(""));
                        let dsp = io::DSP::new(r.info().frame, r.info().rate, &vf2);
                        let w = io::PAWriter::new(
                            output_dev,
                            r.info().frame,
                            r.info().rate,
                            r.info().output_ch,
                        );
                        let r = Box::new(r) as Box<dyn io::Input + Send>;
                        let w = Box::new(w) as Box<dyn io::Output + Send>;
                        let dsp = Box::new(dsp) as Box<dyn io::Processor + Send>;
                        return kani::Kani::start(r, w, dsp).unwrap();
                    }
                } else if cmd == 9 {
                    if input_dev == CLI_DEV_INVALID || output_dev == CLI_DEV_INVALID {
                        log::error!("please select devices first");
                        continue;
                    }
                    let r = io::PAReader::new(input_dev, frame, 48000, 2);
                    let vf2 = std::fs::read_to_string(PATH_FILTERS).unwrap_or(String::from(""));
                    let dsp = io::DSP::new(r.info().frame, r.info().rate, &vf2);
                    let w = io::PAWriter::new(
                        output_dev,
                        r.info().frame,
                        r.info().rate,
                        r.info().output_ch,
                    );
                    let r = Box::new(r) as Box<dyn io::Input + Send>;
                    let w = Box::new(w) as Box<dyn io::Output + Send>;
                    let dsp = Box::new(dsp) as Box<dyn io::Processor + Send>;
                    return kani::Kani::start(r, w, dsp).unwrap();
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
