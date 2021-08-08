use crate::err::DeqError;
use deq_filter as f;
use deq_filter::Convolver;
use deq_filter::Delay;
use deq_filter::Filter;
use deq_filter::{BQFParam, BQFType, BiquadFilter};
use deq_filter::{VocalRemover, VocalRemoverType};
use deq_filter::{Volume, VolumeCurve};
use portaudio as pa;
use std::{thread, time};

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

fn setup_filters(fs: f32) -> (Vec<Box<dyn f::Filter>>, Vec<Box<dyn f::Filter>>) {
    // init filters
    let lfs: Vec<Box<dyn f::Filter>> = vec![
        // BiquadFilter::newb(BQFType::HighPass, fs, 250.0, 0.0, BQFParam::Q(0.707)),
        // BiquadFilter::newb(BQFType::LowPass, fs, 8000.0, 0.0, BQFParam::Q(0.707)),
        // BiquadFilter::newb(BQFType::PeakingEQ, fs, 880.0, 9.0, BQFParam::BW(1.0)),
        Volume::newb(VolumeCurve::Gain, -12.0),
        // Delay::newb(200, fs as usize),
    ];
    let rfs: Vec<Box<dyn f::Filter>> = vec![
        // BiquadFilter::newb(BQFType::HighPass, fs, 250.0, 0.0, BQFParam::Q(0.707)),
        // BiquadFilter::newb(BQFType::LowPass, fs, 8000.0, 0.0, BQFParam::Q(0.707)),
        // BiquadFilter::newb(BQFType::PeakingEQ, fs, 880.0, 9.0, BQFParam::BW(1.0)),
        Volume::newb(VolumeCurve::Gain, -12.0),
        // Delay::newb(200, fs as usize),
    ];
    (lfs, rfs)
}

fn setup_filters2(fs: f32) -> Vec<Box<dyn f::Filter2ch>> {
    let sfs: Vec<Box<dyn f::Filter2ch>> = vec![
        // VocalRemover::newb(VocalRemoverType::RemoveCenter),
        // VocalRemover::newb(VocalRemoverType::RemoveCenterBW(fs, f32::MIN, f32::MAX)),
        VocalRemover::newb(VocalRemoverType::RemoveCenterBW(fs, 200.0, 4800.0)),
    ];
    sfs
}

fn apply_filters(
    l: &[f32],
    r: &[f32],
    lfs: &mut Vec<Box<dyn f::Filter>>,
    rfs: &mut Vec<Box<dyn f::Filter>>,
) -> (Vec<f32>, Vec<f32>) {
    let l = lfs.iter_mut().fold(l.to_vec(), |x, f| f.apply(&x));
    let r = rfs.iter_mut().fold(r.to_vec(), |x, f| f.apply(&x));
    (l, r)
}

fn apply_filters2(
    l: &[f32],
    r: &[f32],
    sfs: &mut Vec<Box<dyn f::Filter2ch>>,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(l.len(), r.len());
    let debug = l.len();

    // this returns wrong length vecs
    // let (l, r) = sfs
    //     .iter_mut()
    //     .fold((l.to_vec(), r.to_vec()), |(l, r), f| f.apply(&l, &r));

    let mut l = l.to_vec();
    let mut r = r.to_vec();
    // compile error "destructuring assignments are unstable"
    // (l2, r2) = sfs[0].apply(&l, &r);
    let (l2, r2) = sfs[0].apply(&l, &r);
    l = l2;
    r = r2;

    assert_eq!(l.len(), r.len());
    assert_eq!(l.len(), debug);
    (l, r)
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
    let rbuf: Vec<i16> = reader.samples().map(|s| s.unwrap()).collect();

    // init filters
    let fs = reader.spec().sample_rate as f32;
    let (mut lfs, mut rfs) = setup_filters(fs);
    let mut sfs = setup_filters2(fs);

    // pa stream
    let ch = 2;
    let frame: u32 = 1024;
    let interleaved = true;

    // apply filters
    let uframe = frame as usize * ch as usize;
    let mut rbuf = f::i16_to_f32(&rbuf);
    let zero_paddings = uframe - (rbuf.len() as usize % uframe);
    let zeros = vec![0.0; zero_paddings];
    rbuf.extend(zeros);
    assert_eq!(rbuf.len() % uframe, 0);
    let mut buf = Vec::with_capacity(rbuf.len());
    let loopnum = rbuf.len() / uframe;
    for i in 0..loopnum {
        let a = i * uframe;
        let b = (i + 1) * uframe;
        let tmp = rbuf[a..b].to_vec();
        assert_eq!(tmp.len(), uframe);
        let (l, r) = f::from_interleaved(&tmp);
        let (l, r) = apply_filters(&l, &r, &mut lfs, &mut rfs);
        let (l, r) = apply_filters2(&l, &r, &mut sfs);
        let tmp = f::to_interleaved(&l, &r);
        assert_eq!(tmp.len(), uframe);
        buf.extend(tmp);
    }
    assert_eq!(buf.len(), rbuf.len());
    let mut buf = buf.into_iter();

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
                    Some(s) => s,
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

    // init filters
    let fs = rate as f32;
    let (mut lfs, mut rfs) = setup_filters(fs);
    let mut sfs = setup_filters2(fs);

    // // use thread to process signals (1/3)
    // let (send_lx, recv_lx) = ::std::sync::mpsc::channel();
    // let (send_rx, recv_rx) = ::std::sync::mpsc::channel();
    // let (send_ly, recv_ly) = ::std::sync::mpsc::channel();
    // let (send_ry, recv_ry) = ::std::sync::mpsc::channel();
    // let hdl_l = thread::spawn(move || {
    //     let ir = f::load_ir(&std::fs::read_to_string("ir").expect(""), CONV_N);
    //     let mut lfs: Vec<Box<dyn f::Filter>> = vec![
    //         // BiquadFilter::newb(BQFType::HighPass, fs, 250.0, 0.0, BQFParam::Q(0.707)),
    //         // BiquadFilter::newb(BQFType::LowPass, fs, 8000.0, 0.0, BQFParam::Q(0.707)),
    //         // BiquadFilter::newb(BQFType::PeakingEQ, fs, 880.0, 9.0, BQFParam::BW(1.0)),
    //         Convolver::newb(&ir),
    //         Volume::newb(VolumeCurve::Gain, -12.0),
    //         // Delay::newb(200, fs as usize),
    //     ];
    //     // TODO: close when stop
    //     loop {
    //         let l = recv_lx.recv().unwrap();
    //         let l = lfs.iter_mut().fold(l, |x, f| f.apply(x));
    //         send_ly.send(l).ok();
    //     }
    // });
    // let hdl_r = thread::spawn(move || {
    //     let ir = f::load_ir(&std::fs::read_to_string("ir").expect(""), CONV_N);
    //     let mut rfs: Vec<Box<dyn f::Filter>> = vec![
    //         // BiquadFilter::newb(BQFType::HighPass, fs, 250.0, 0.0, BQFParam::Q(0.707)),
    //         // BiquadFilter::newb(BQFType::LowPass, fs, 8000.0, 0.0, BQFParam::Q(0.707)),
    //         // BiquadFilter::newb(BQFType::PeakingEQ, fs, 880.0, 9.0, BQFParam::BW(1.0)),
    //         Convolver::newb(&ir),
    //         Volume::newb(VolumeCurve::Gain, -12.0),
    //         // Delay::newb(200, fs as usize),
    //     ];
    //     // TODO: close when stop
    //     loop {
    //         let r = recv_rx.recv().unwrap();
    //         let r = rfs.iter_mut().fold(r, |x, f| f.apply(x));
    //         send_ry.send(r).ok();
    //     }
    // });

    // latency channel
    // exit if received u128::MAX
    let (sender, receiver) = ::std::sync::mpsc::channel();

    // setup callback function
    let callback = move |pa::DuplexStreamCallbackArgs {
                             in_buffer,
                             out_buffer,
                             ..
                         }| {
        let start = time::Instant::now();
        // --- measure latency ---
        assert_eq!(in_buffer.len(), frame as usize * 2);
        let (l, r) = f::from_interleaved(in_buffer);
        // apply filters
        let (l, r) = apply_filters(&l, &r, &mut lfs, &mut rfs);
        let (l, r) = apply_filters2(&l, &r, &mut sfs);
        let buf = f::to_interleaved(&l, &r);
        assert_eq!(buf.len(), frame as usize * 2);
        // // use thread to process signals (2/3)
        // send_lx.send(l).ok();
        // send_rx.send(r).ok();
        // let l = recv_ly.recv().unwrap();
        // let r = recv_ry.recv().unwrap();
        for (o_sample, i_sample) in out_buffer.iter_mut().zip(buf.iter()) {
            *o_sample = *i_sample;
        }
        // -----------------------
        let end = start.elapsed();
        sender.send(end.as_micros()).ok();

        // TODO
        if true {
            pa::Continue
        } else {
            sender.send(std::u128::MAX).ok();
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
    let mut latency_avg = 0.0f64;
    let avg_sec = 5;
    let n = rate as u64 / frame as u64 * avg_sec;
    let mut count: u64 = 0;
    loop {
        // stats
        let l = receiver.recv().unwrap();
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
        // stop?
        if l == std::u128::MAX {
            log::debug!("stop");
            break;
        }
    }

    // // use thread to process signals (3/3)
    // hdl_l.join().unwrap();
    // hdl_r.join().unwrap();

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
