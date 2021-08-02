use crate::err::DeqError;
use portaudio as pa;

pub fn get_device_names(pa: &pa::PortAudio) -> Result<Vec<(usize, String)>, DeqError> {
    log::trace!("get devices");
    let devs = pa.devices();
    match devs {
        Ok(devs) => Ok(devs
            .map(|dev| {
                let d = dev.unwrap();
                let pa::DeviceIndex(idx) = d.0;
                (idx as usize, String::from(d.1.name))
            })
            .collect()),
        Err(_) => Err(DeqError::Device),
    }
}

pub fn get_device_info(pa: &pa::PortAudio, idx: usize) -> Result<pa::DeviceInfo, DeqError> {
    let devidx = pa::DeviceIndex(idx as u32);
    log::trace!("get devices");
    let devs = pa.devices();
    if let Ok(devs) = devs {
        for d in devs {
            if let Ok(d) = d {
                if d.0 == devidx {
                    return Ok(d.1);
                }
            } else {
                return Err(DeqError::Device);
            }
        }
    }
    Err(DeqError::Device)
}

pub fn play_wav(pa: &pa::PortAudio, path: &str, dev: usize) -> Result<(), DeqError> {
    // read wav
    log::debug!("read wav {}", path);
    let reader = hound::WavReader::open(path);
    if let Err(e) = reader {
        log::error!("could not open wav err={}", e);
        return Err(DeqError::Format);
    }
    let mut reader = reader.unwrap();
    log::debug!("wav_info={:?}", reader.spec());
    if reader.spec().sample_format != hound::SampleFormat::Int
        || reader.spec().bits_per_sample != 16
        || reader.spec().channels != 2
    {
        log::debug!("wav must be 2ch pcm_s16le");
        return Err(DeqError::Format);
    }
    let buf: Vec<i16> = reader.samples().map(|s| s.unwrap()).collect();
    let mut buf = buf.iter();

    // pa stream
    let ch = 2;
    let frame = 1024;
    let interleaved = true;
    let player_info = get_device_info(pa, dev)?;
    // let rate = player_info.default_sample_rate;
    let rate = reader.spec().sample_rate as f64; // try wav's rate
    let latency = player_info.default_low_output_latency;
    let stream_params =
        pa::StreamParameters::<i16>::new(pa::DeviceIndex(dev as u32), ch, interleaved, latency);
    if let Err(e) = pa.is_output_format_supported(stream_params, rate) {
        log::error!("format not supported err={}", e);
        return Err(DeqError::Format);
    }
    let stream_settings = pa::OutputStreamSettings::new(stream_params, rate, frame);

    log::trace!("open stream");
    let stream = pa.open_blocking_stream(stream_settings);
    if let Err(e) = stream {
        log::error!("could not open stream err={}", e);
        return Err(DeqError::Format);
    }
    let mut stream = stream.unwrap();
    if let Err(e) = stream.start() {
        log::error!("could not start stream err={}", e);
        return Err(DeqError::Format);
    }
    log::debug!("stream started info={:?}", stream.info());

    // filter func
    let filter_func = |x: i16| {
        let volume_max = 100;
        let volume = 50;
        (x as i64 * volume) / volume_max
    };

    // playback
    let mut is_finished = false;
    let num_samples = (frame * ch as u32) as usize;
    loop {
        if is_finished {
            log::debug!("finish play stream");
            break;
        }
        // write one frame
        if let Err(e) = stream.write(frame, |w| {
            for i in 0..num_samples {
                let sample = match buf.next() {
                    Some(s) => *s,
                    None => {
                        if !is_finished {
                            log::debug!("wav finished");
                            is_finished = true;
                        }
                        0 // silent
                    }
                };
                let filtered = filter_func(sample);
                log::trace!("write i={} sample={} filtered={}", i, sample, filtered);
                w[i] = filtered as i16;
            }
        }) {
            log::warn!("play stream err={}", e);
        };
    }
    Ok(())
}
