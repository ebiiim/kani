# ♪

**_Work In Progress_**

A toy audio processor I made in my holidays just for fun :D

## Basic Concepts

```
    Input          Configurable Audio Processor          Output
┌───────────┐      ┌──────┬──────┬──────┬──────┐      ┌───────────┐
│           │ L ch │      │      │      │      │ L ch │           │
│           │  ┌──►│ LPF  │  EQ  │  EQ  │ Gain ├──┐   │           │
│           │  │   │      │      │      │      │  │   │           │
│  Line In  ├──┤   ├──────┼──────┼──────┼──────┤  ├──►│ Line Out  │
│           │  │   │      │      │      │      │  │   │           │
│           │  └──►│ LPF  │ None │  EQ  │ Gain ├──┘   │           │
│           │ R ch │      │      │      │      │ R ch │           │
└───────────┘      └──────┴──────┴──────┴──────┘      └───────────┘
```


## Features

- DSP
    - [x] Volume
    - [x] Parametric Equalizer (Biquad Filter)
        - Low Pass / High Pass / Peak EQ / High Shelf / Low Shelf
    - [x] Vocal Remover (just might work)
    - [x] Convolver (super slow toy implementation)
    - [ ] FFT
- Input / Output
    - [x] Audio Interfaces (using PortAudio, ~200ms latency)
    - [x] Files (currently 2ch 16bit WAVE only)
    - [x] Pipe
    - [x] Spotify Connect (using librespot & pipe)
- User Interfaces
    - [x] CLI (minimal one)
    - [ ] Config (Import / Export)
    - [ ] GUI
- Supported Platforms (should work on same platforms as PortAudio)
    - [x] Windows
    - [x] Linux
    - [x] macOS

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

### Compile

```sh
cargo build --release
```
