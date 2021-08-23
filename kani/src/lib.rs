use anyhow::{bail, Context, Result};
use io::Status::*;
use kani_io as io;
use serde::{Deserialize, Serialize};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;

#[derive(Debug, Default, Copy, Clone, Serialize, Deserialize)]
pub struct PlayerInfo {
    pub frame_size: u32,
    pub sampling_rate: u32,
}

#[derive(Debug, Default, Copy, Clone, Serialize, Deserialize)]
pub struct CurrentStatus {
    pub frames: u64,
    pub latency_us: u32,
    pub avg_latency_us: u32,

    pub in_l_rms: f32,
    pub in_l_peak: f32,
    pub in_r_rms: f32,
    pub in_r_peak: f32,

    pub out_l_rms: f32,
    pub out_l_peak: f32,
    pub out_r_rms: f32,
    pub out_r_peak: f32,
}

#[derive(Debug)]
pub struct Kani {
    info: Mutex<PlayerInfo>,
    status: Mutex<CurrentStatus>,
    filters: Mutex<String>,

    in_cmd_tx: SyncSender<io::Cmd>,
    out_cmd_tx: SyncSender<io::Cmd>,
    dsp_cmd_tx: SyncSender<io::Cmd>,
}

impl Kani {
    const AVG_SEC: u32 = 2;

    pub fn start(
        mut r: Box<dyn io::Input + Send>,
        mut w: Box<dyn io::Output + Send>,
        mut dsp: Box<dyn io::Processor + Send>,
    ) -> Result<Arc<Self>> {
        log::debug!(
            "Kani::start with r={:?} dsp={:?} w={:?}",
            r.info(),
            dsp.info(),
            w.info()
        );
        // use sync channel to pace the reader so do not use async channel
        // use small buffer and let channels no rendezvous
        let (in_tx, dsp_rx) = sync_channel(1);
        let (dsp_tx, out_rx) = sync_channel(1);

        // please receive!
        let (in_status_tx, status_rx) = sync_channel(0);
        let dsp_status_tx = in_status_tx.clone();
        let out_status_tx = in_status_tx.clone();

        // receivers do try_recv() so use try_send and let buffer>0
        let (in_cmd_tx, in_cmd_rx) = sync_channel(1);
        let (dsp_cmd_tx, dsp_cmd_rx) = sync_channel(1);
        let (out_cmd_tx, out_cmd_rx) = sync_channel(1);

        let info = PlayerInfo {
            frame_size: w.info().input_frame as u32,
            sampling_rate: w.info().input_rate as u32,
        };
        // frames per sec.
        let n = info.sampling_rate as u32 / info.frame_size as u32 * Self::AVG_SEC;

        let _ = thread::spawn(move || {
            r.run(in_tx, in_status_tx, in_cmd_rx).unwrap();
        });
        let _ = thread::spawn(move || {
            dsp.run(dsp_rx, dsp_tx, dsp_status_tx, dsp_cmd_rx).unwrap();
        });
        let _ = thread::spawn(move || {
            w.run(out_rx, out_status_tx, out_cmd_rx).unwrap();
        });

        let p = Self {
            in_cmd_tx,
            dsp_cmd_tx,
            out_cmd_tx,
            filters: Mutex::new(String::from("[]")),
            info: Mutex::new(info),
            status: Mutex::new(CurrentStatus {
                ..Default::default()
            }),
        };

        let p = Arc::new(p);
        let p2 = p.clone(); // return this

        // wait for TxInit/RxInit
        status_rx.recv().unwrap();
        status_rx.recv().unwrap();
        status_rx.recv().unwrap();
        status_rx.recv().unwrap();
        // synchronize
        p.out_cmd_tx.send(io::Cmd::Start).unwrap();
        p.dsp_cmd_tx.send(io::Cmd::Start).unwrap();
        p.in_cmd_tx.send(io::Cmd::Start).unwrap();

        let _ = thread::spawn(move || {
            for s in status_rx {
                match s {
                    // check io::Status::*
                    Latency(l) => {
                        let mut ps = p.status.lock().unwrap();
                        // count frames
                        ps.frames += 1;

                        // calc latency
                        ps.latency_us = l;
                        ps.avg_latency_us -= ps.avg_latency_us / n;
                        ps.avg_latency_us += l / n;

                        // once every avg_sec
                        if ps.frames % n as u64 == 0 {
                            log::debug!(
                                "DSP latency ({}s avg): {:.3} ms | total frames: {}",
                                Self::AVG_SEC,
                                ps.avg_latency_us as f32 / 1000.,
                                ps.frames
                            );
                        }
                    }
                    RMS(pos, ch, v) => {
                        let mut ps = p.status.lock().unwrap();

                        if pos == io::IO::Input {
                            if ch == 0 {
                                ps.in_l_rms = v;
                            } else {
                                ps.in_r_rms = v;
                            }
                        } else {
                            if ch == 0 {
                                ps.out_l_rms = v;
                            } else {
                                ps.out_r_rms = v;
                            }
                        }
                    }
                    Peak(pos, ch, v) => {
                        let mut ps = p.status.lock().unwrap();

                        if pos == io::IO::Input {
                            if ch == 0 {
                                ps.in_l_peak = v;
                            } else {
                                ps.in_r_peak = v;
                            }
                        } else {
                            if ch == 0 {
                                ps.out_l_peak = v;
                            } else {
                                ps.out_r_peak = v;
                            }
                        }
                    }
                    Loaded(j) => {
                        log::info!("Cmd::Loaded",);
                        let mut pf = p.filters.lock().unwrap();
                        *pf = j;
                    }
                    Interpolated(_) => {
                        log::info!("{:?}", s);
                    }
                    TxInit(_) | RxInit(_) => {
                        log::info!("{:?}", s)
                    }
                    TxFin(_) | RxFin(_) => {
                        log::info!("{:?}", s)
                    }
                    TxErr(_) | RxErr(_) => {
                        log::warn!("{:?}", s)
                    }
                    _ => {
                        log::trace!("{:?}", s)
                    }
                }
            }
        });

        Ok(p2)
    }
    pub fn status(&self) -> CurrentStatus {
        let s = self.status.lock().unwrap();
        s.clone()
    }
    pub fn info(&self) -> PlayerInfo {
        let i = self.info.lock().unwrap();
        i.clone()
    }
    pub fn filters(&self) -> String {
        let f = self.filters.lock().unwrap();
        f.clone()
    }
    pub fn set_filters(&self, vf2: &str) -> Result<()> {
        log::info!("reload filters");
        if let Err(e) = self.dsp_cmd_tx.try_send(io::Cmd::Reload(vf2.to_string())) {
            bail!("could not reload filters as {}", e);
        }
        Ok(())
    }
    pub fn stop(&self) -> Result<()> {
        if self.in_cmd_tx.try_send(io::Cmd::Stop).is_err()
            || self.dsp_cmd_tx.try_send(io::Cmd::Stop).is_err()
            || self.out_cmd_tx.try_send(io::Cmd::Stop).is_err()
        {
            log::info!("could not send stop (already stopped?)");
        }
        Ok(())
    }
}
