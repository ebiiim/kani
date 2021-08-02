use portaudio;

mod cli;
mod err;

fn main() {
    let pa = portaudio::PortAudio::new().unwrap();
    let (input_dev, output_dev) = cli::select_devices_loop(&pa);
    println!("Input device: {}, Output device: {}", input_dev, output_dev);
}
