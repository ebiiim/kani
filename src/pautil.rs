use crate::err::DeqError;
use crate::filter;
use crate::filter::Delay;
use crate::filter::{BQFParam, BQFType, BiquadFilter};
use crate::filter::{Volume, VolumeCurve};

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

    // apply filters
    let fs = reader.spec().sample_rate as f32;
    let l1 = BiquadFilter::new(BQFType::HighPass, fs, 250.0, 0.0, BQFParam::Q(0.707));
    let r1 = BiquadFilter::new(BQFType::HighPass, fs, 250.0, 0.0, BQFParam::Q(0.707));
    let l2 = BiquadFilter::new(BQFType::LowPass, fs, 8000.0, 0.0, BQFParam::Q(0.707));
    let r2 = BiquadFilter::new(BQFType::LowPass, fs, 8000.0, 0.0, BQFParam::Q(0.707));
    let l3 = BiquadFilter::new(BQFType::PeakingEQ, fs, 880.0, 9.0, BQFParam::BW(1.0));
    let r3 = BiquadFilter::new(BQFType::PeakingEQ, fs, 880.0, 9.0, BQFParam::BW(1.0));
    let lv = Volume::new(VolumeCurve::Linear, 0.2);
    let rv = Volume::new(VolumeCurve::Linear, 0.2);
    let ld = Delay::new(200, fs as usize);
    let rd = Delay::new(200, fs as usize);

    let buf = filter::i16_to_f32(buf);
    let (l, r) = filter::from_interleaved(buf);
    let l = filter::apply(l1, l);
    let r = filter::apply(r1, r);
    let l = filter::apply(l2, l);
    let r = filter::apply(r2, r);
    let l = filter::apply(l3, l);
    let r = filter::apply(r3, r);
    let l = filter::apply(lv, l);
    let r = filter::apply(rv, r);
    let l = filter::apply(ld, l);
    let r = filter::apply(rd, r);
    let buf = filter::to_interleaved(l, r);
    let mut buf = buf.iter();

    // pa stream
    let ch = 2;
    let frame = 1024;
    let interleaved = true;
    let player_info = get_device_info(pa, dev)?;
    // let rate = player_info.default_sample_rate;
    let rate = fs as f64; // try wav's rate instead of default_sample_rate
    let latency = player_info.default_low_output_latency;
    // PortAudio supports audio in/out in f32 regardless of the native audio API.
    // Applying filters in floating point avoids precision loss, so use f32 output here to reduce conversions.
    let stream_params =
        pa::StreamParameters::<f32>::new(pa::DeviceIndex(dev as u32), ch, interleaved, latency);
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
            #[allow(clippy::needless_range_loop)]
            for i in 0..num_samples {
                let sample = match buf.next() {
                    Some(s) => *s,
                    None => {
                        if !is_finished {
                            log::debug!("wav finished");
                            is_finished = true;
                        }
                        0.0 // silent
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

pub fn play(pa: &pa::PortAudio, i_dev: usize, o_dev: usize) -> Result<(), DeqError> {
    let ch = 2;
    let frame = 1024;
    let interleaved = true;
    let i_dev_info = get_device_info(pa, i_dev)?;
    let o_dev_info = get_device_info(pa, o_dev)?;
    let rate;
    #[allow(clippy::float_cmp)]
    if i_dev_info.default_sample_rate == o_dev_info.default_sample_rate {
        rate = i_dev_info.default_sample_rate;
    } else {
        rate = 48000.0; // most devices are compatible with this
    }
    let i_latency = i_dev_info.default_low_output_latency;
    let o_latency = o_dev_info.default_low_output_latency;
    let i_params =
        pa::StreamParameters::<f32>::new(pa::DeviceIndex(i_dev as u32), ch, interleaved, i_latency);
    let o_params =
        pa::StreamParameters::<f32>::new(pa::DeviceIndex(o_dev as u32), ch, interleaved, o_latency);
    if let Err(e) = pa.is_duplex_format_supported(i_params, o_params, rate) {
        log::error!("format not supported err={}", e);
        return Err(DeqError::Format);
    }
    let settings = pa::DuplexStreamSettings::new(i_params, o_params, rate, frame);

    // count total frames; exit if received -1
    let (sender, receiver) = ::std::sync::mpsc::channel();
    let mut count: i64 = 0;

    // apply filters
    let fs = rate as f32;
    let mut l1 = BiquadFilter::new(BQFType::HighPass, fs, 250.0, 0.0, BQFParam::Q(0.707));
    let mut r1 = BiquadFilter::new(BQFType::HighPass, fs, 250.0, 0.0, BQFParam::Q(0.707));
    let mut l2 = BiquadFilter::new(BQFType::LowPass, fs, 8000.0, 0.0, BQFParam::Q(0.707));
    let mut r2 = BiquadFilter::new(BQFType::LowPass, fs, 8000.0, 0.0, BQFParam::Q(0.707));
    let mut l3 = BiquadFilter::new(BQFType::PeakingEQ, fs, 880.0, 9.0, BQFParam::BW(1.0));
    let mut r3 = BiquadFilter::new(BQFType::PeakingEQ, fs, 880.0, 9.0, BQFParam::BW(1.0));
    let mut lv = Volume::new(VolumeCurve::Linear, 0.2);
    let mut rv = Volume::new(VolumeCurve::Linear, 0.2);
    let mut ld = Delay::new(200, fs as usize);
    let mut rd = Delay::new(200, fs as usize);

    // setup callback function
    let callback = move |pa::DuplexStreamCallbackArgs {
                             in_buffer,
                             out_buffer,
                             ..
                         }| {
        let (l, r) = filter::from_interleaved(in_buffer.to_vec());
        let l = l1.apply(l);
        let r = r1.apply(r);
        let l = l2.apply(l);
        let r = r2.apply(r);
        let l = l3.apply(l);
        let r = r3.apply(r);
        let l = lv.apply(l);
        let r = rv.apply(r);
        let l = ld.apply(l);
        let r = rd.apply(r);
        let buf = filter::to_interleaved(l, r);

        for (o_sample, i_sample) in out_buffer.iter_mut().zip(buf.iter()) {
            *o_sample = *i_sample;
        }

        sender.send(count).ok();
        count += 1;

        // TODO
        if true {
            pa::Continue
        } else {
            sender.send(-1).ok();
            pa::Complete
        }
    };

    log::trace!("open a non-blocking stream");
    let stream = pa.open_non_blocking_stream(settings, callback);
    if stream.is_err() {
        log::error!("failed to open the stream");
        return Err(DeqError::Format);
    }
    let mut stream = stream.unwrap();
    if let Err(e) = stream.start() {
        log::error!("failed to open stream err={:?}", e);
        return Err(DeqError::Format);
    }

    // block
    loop {
        let c = receiver.recv().unwrap_or(0);
        log::trace!("frame {:?}", c);
        if c == -1 {
            log::debug!("stop");
            break;
        }
    }
    if let Err(e) = stream.stop() {
        log::warn!("could not stop playback stream err={}", e);
    }
    loop {
        let cont = stream.is_active();
        if let Err(e) = cont {
            log::warn!("stream.is_active() err={:?}", e);
        }
        if !cont.unwrap() {
            break;
        }
    }

    Ok(())
}
