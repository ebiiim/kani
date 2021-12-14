# kani🦀

<sup>"kani" means club in Japanese</sup>

Toy audio processor I made in my holidays for fun :D

![docs/demo.gif](https://raw.githubusercontent.com/ebiiim/kani/main/docs/demo.gif)

## Design

```
    Input          Configurable Audio Processor          Output
┌───────────┐      ┌──────┬──────┬──────┬──────┐      ┌───────────┐
│           │ L ch │      │      │      │      │ L ch │           │
│           │  ┌──►│  LPF │  EQ  │  EQ  │      ├──┐   │           │
│           │  │   │      │      │      │ Cross│  │   │           │
│  Line In  ├──┤   ├──────┼──────┼──────┤ feed │  ├──►│ Line Out  │
│           │  │   │      │      │      │      │  │   │           │
│           │  └──►│  LPF │  EQ  │ None │      ├──┘   │           │
│           │ R ch │      │      │      │      │ R ch │           │
└───────────┘      └──────┴──────┴──────┴──────┘      └───────────┘
```


## Features

- DSP
    - [x] Volume
    - [x] Parametric Equalizer (Biquad Filter)
        - Low Pass / High Pass / Peak EQ / High Shelf / Low Shelf
    - [x] Delay
    - [x] Crossfeed (just might work)
    - [x] Vocal Remover (just might work)
    - [x] Convolver (super slow non-FFT implementation)
    - [ ] FFT
    - [x] Normalizer (ajusts volume when detects signal > 1.0)
- Input / Output
    - [x] Audio Interfaces (using PortAudio, ~200ms latency)
    - [x] Files (currently 2ch 16bit WAVE only)
    - [x] Pipe
    - [x] Spotify Connect (using librespot & pipe)
    - [x] Resampling
      - [ ] Cool interpolation instead of copy and paste
- User Interfaces
    - [x] CLI
      - [x] with cool RMS meter🥰
    - [ ] GUI
    - [x] Config (Import / Export)
      - [x] Update DSP config during playback
- Supported Platforms (should work on same platforms as PortAudio)
    - [x] Linux
      - [x] ARMv7
    - [x] macOS
    - [x] Windows
      - [ ] ASIO

## Build

### Prerequisites

[PortAudio](http://www.portaudio.com/)

```sh
# Linux
sudo apt install portaudio19-dev
# macOS
brew install portaudio
# Windows
.\vcpkg install portaudio
```

[librespot](https://github.com/librespot-org/librespot)

```sh
# Linux only
sudo apt-get install build-essential libasound2-dev
# All platforms
cargo install librespot
```

### Run

```sh
git clone https://github.com/ebiiim/kani && cd kani
cp ~/.cargo/bin/librespot{,.exe} .
cargo run --release
```
