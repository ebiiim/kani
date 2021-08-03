pub fn i16_to_f32(a: Vec<i16>) -> Vec<f32> {
    log::trace!("i16_to_f32 a.len()={}", a.len());
    a.iter().map(|x| *x as f32 / 32767.0).collect()
}

pub fn from_interleaved(a: Vec<f32>) -> (Vec<f32>, Vec<f32>) {
    log::trace!("from_interleaved a.len()={}", a.len());
    let mut l: Vec<f32> = vec![0.0; a.len() / 2];
    let mut r: Vec<f32> = vec![0.0; a.len() / 2];
    for i in 0..a.len() {
        if i % 2 == 0 {
            l[i / 2] = a[i];
        } else {
            r[i / 2] = a[i];
        }
    }
    (l, r)
}

pub fn to_interleaved(l: Vec<f32>, r: Vec<f32>) -> Vec<f32> {
    log::trace!("from_interleaved l.len()={} r.len()={}", l.len(), r.len());
    if l.len() != r.len() {
        log::error!(
            "channel buffers must have same length but l={} r={}",
            l.len(),
            r.len()
        );
        return Vec::new();
    }
    let mut a: Vec<f32> = vec![0.0; l.len() * 2];
    for i in 0..a.len() {
        if i % 2 == 0 {
            a[i] = l[i / 2];
        } else {
            a[i] = r[i / 2];
        }
    }
    a
}

pub fn volume(a: Vec<f32>, vol: f32) -> Vec<f32> {
    log::trace!("volume a.len()={} vol={}", a.len(), vol,);
    a.iter().map(|x| *x * vol).collect()
}
