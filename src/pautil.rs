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

pub fn from_interleaved(a: &[i16]) -> (Vec<i16>, Vec<i16>) {
    log::trace!("from_interleaved a.len()={}", a.len());
    let mut l: Vec<i16> = vec![0; a.len() / 2];
    let mut r: Vec<i16> = vec![0; a.len() / 2];
    for i in 0..a.len() {
        if i % 2 == 0 {
            l[i / 2] = a[i];
        } else {
            r[i / 2] = a[i];
        }
    }
    (l, r)
}

pub fn to_interleaved(l: &[i16], r: &[i16]) -> Vec<i16> {
    log::trace!("from_interleaved l.len()={} r.len()={}", l.len(), r.len());
    if l.len() != r.len() {
        log::error!(
            "channel buffers must have same length but l={} r={}",
            l.len(),
            r.len()
        );
        return Vec::new();
    }
    let mut a: Vec<i16> = vec![0; l.len() * 2];
    for i in 0..a.len() {
        if i % 2 == 0 {
            a[i] = l[i / 2];
        } else {
            a[i] = r[i / 2];
        }
    }
    a
}

pub fn volume_filter(a: &[i16], vol: isize, vol_max: isize) -> Vec<i16> {
    log::trace!(
        "volume_filter a.len()={} vol={} vol_max={}",
        a.len(),
        vol,
        vol_max
    );
    // Process audio with at least 32 bits (or 64 bits on 64-bit arch)
    a.iter()
        .map(|x| ((*x as isize * vol) / vol_max) as i16)
        .collect()
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

    // apply filters
    let (l, r) = from_interleaved(&buf);
    let l = volume_filter(&l, 10, 100);
    let r = volume_filter(&r, 10, 100);
    let buf = to_interleaved(&l, &r);

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

    // playback
    let mut is_finished = false;
    let num_samples = (frame * ch as u32) as usize;
    loop {
        if is_finished {
            log::debug!("finish playback stream");
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
                w[i] = sample;
            }
        }) {
            log::warn!("playback stream err={}", e);
        };
    }
    if let Err(e) = stream.stop() {
        log::warn!("could not stop playback stream err={}", e);
    }
    Ok(())
}
