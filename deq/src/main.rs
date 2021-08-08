use portaudio as pa;
use std::env;
use std::process;

extern crate deq_filter;
extern crate deq_io;

mod cli;
mod err;
mod pautil;

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

    // init portaudio
    let pa = pa::PortAudio::new();
    if pa.is_err() {
        log::error!("could not initialize portaudio: {:?}", pa.unwrap_err());
        process::exit(0);
    }
    let pa = pa.unwrap();

    cli::start(&pa);
}
