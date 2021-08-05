#!/usr/bin/env python

import sys
from typing import List, Tuple
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.numeric import convolve
from scipy import signal
from functools import reduce


def _plot(hz, gain, phase=None, title="no title", show=True, save=True, ext="png"):
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(111)
    ax1.set_title(title)
    ax1.set_ylabel("Gain (dB)")
    # -----
    # ax1.set_ylim(-18, 18)
    # ax1.set_yticks([-18, -15, -12, -9, -6, -3,
    #                 0, 3, 6, 9, 12, 15, 18])
    # ax1.set_yticklabels(["", "-15", "", "-9", "", "-3",
    #                     "0", "3", "", "9", "", "15", ""])
    # -----
    ax1.set_ylim(-12, 12)
    ax1.set_yticks([-12, -9, -6, -3, 0, 3, 6, 9, 12])
    # -----
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_xlim(20, 20000)
    ax1.set_xscale("log")
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_xticks([20, 30, 40, 50, 60, 70, 80, 90, 100,
                    200, 300, 400, 500, 600, 700, 800, 900, 1000,
                    2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
                    20000])
    ax1.set_xticklabels(["20", "30", "40", "", "60", "", "80", "", "100",
                         "200", "", "400", "", "", "", "800", "", "1k",
                        "2k", "3k", "4k", "", "6k", "", "8k", "", "10k",
                         "20k"])
    ax1.grid()

    if phase is not None:
        ax2 = ax1.twinx()
        ax2.set_ylabel("Phase (deg.)")
        ax2.set_ylim(-180, 180)
        # -----
        # ax2.set_yticks([-180, -150, -120, -90, -60, -30,
        #                 0, 30, 60, 90, 120, 150, 180])
        # ax2.set_yticklabels(["-180", "", "-120", "", "-60", "",
        #                      "0", "", "60", "", "120", "", "180"])
        # -----
        ax2.set_yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
        # -----

    # plot
    if phase is not None:
        ax2.plot(hz, phase, "cyan")
    ax1.plot(hz, gain, "blue")

    if save:
        plt.savefig(title+"."+ext)

    if show:
        plt.show()


def plot_from_coeff(b, a, n, title="no title", fs=48000.0, show=True, save=True, ext="png"):
    w, h = signal.freqz(b, a, worN=n)
    mag = np.abs(h)
    gain = 20*np.log10(mag/1.0)
    phase = np.angle(h)
    phase_deg = phase*180/math.pi
    hz = fs/2 * w/math.pi  # [0..fs/2]
    _plot(hz, gain, phase_deg, title, show, save, ext)


def plot_from_ir(ir, n, title="no title", fs=48000.0, show=True, save=True, ext="png"):
    h = np.fft.fft(ir, n)
    mag = np.abs(h)
    gain = 20*np.log10(mag/1.0)
    phase = np.angle(h)
    phase_deg = phase*180/math.pi
    hz = np.linspace(0, fs, n, endpoint=False)  # [0..fs]
    _plot(hz, gain, phase_deg, title, show, save, ext)


# returns (result_type, [b], [a], ir)
#
# result_type
#   0: error
#   1: coefficients
#   2: impulse response
#
# [b]
#   [[b0, b1, b2, ...], ...]
#
# [a]
#   [[a0, a1, a2, ...], ...]
#
# ir
#   [t0, t1, t2, ...]
def parse_input(file: str) -> Tuple[int, List[list], List[list], list]:
    buf = []
    if file == "":
        buf = sys.stdin.read().splitlines()
    else:
        with open(file) as f:
            buf = f.read().splitlines()
    if len(buf) == 0:
        return 0, [[]], [[]], []

    result_type = 1
    vb, va = [], []
    ir = []
    if buf[0] not in ["b", "a"]:
        result_type = 2

    # Coeff.
    if result_type == 1:
        tmp = []
        is_a = False
        for x in buf:
            if x in ["b", "a"]:
                if len(tmp) != 0:
                    if is_a:
                        va.append(tmp)
                    else:
                        vb.append(tmp)
                    tmp = []
                if x == "a":
                    is_a = True
                else:
                    is_a = False
                continue
            try:
                tmp.append(float(x))
            except Exception as e:
                print(f"{e}", file=sys.stderr)
                return 0, [[]], [[]], []
        if is_a:
            va.append(tmp)
        else:
            vb.append(tmp)

    # IR
    if result_type == 2:
        try:
            ir = [float(x) for x in buf]
        except Exception as e:
            print(f"{e}", file=sys.stderr)
            return 0, [[]], [[]], []

    return result_type, vb, va, ir


# requires len(v) != 0
def convolve_coeffs(v: List[list]) -> list:
    return reduce(lambda x, y: signal.convolve(x, y), v, [1.0])


def main():
    import argparse
    from datetime import datetime as dt
    parser = argparse.ArgumentParser(
        description="Plot frequency response and phase from coefficients or impulse response.")
    parser.add_argument("-i", dest="file", type=str, nargs=1,
                        default="", help="input file (default: use stdin)")
    parser.add_argument("-t", dest="title",  type=str, nargs=1,
                        default="", help="title (default: {yyyyMMddHHmmss}-{Coeff|IR})")
    parser.add_argument("-r", dest="rate", type=float, nargs=1,
                        default=48000.0, help="sampling rate (default: 48000)")
    parser.add_argument("-n", dest="show",
                        action="store_false", help="no show plot")
    parser.add_argument("-s", dest="save",
                        action="store_true", help="save PNG")
    # parser.add_argument("--dump", dest="dump", action="store_true", help="dump PNG to stdout")

    args = parser.parse_args()
    # print(args, file=sys.stderr)

    is_coeff = False
    file = ""
    if len(args.file) != 0:
        file = args.file[0]
    result_type, vb, va, ir = parse_input(file)
    if result_type == 0:
        print("wrong format", file=sys.stderr)
        exit(1)
    if result_type == 1:
        is_coeff = True
    title = args.title
    if title == "":
        title = dt.now().strftime("%Y%m%d%H%M%S")
        if is_coeff:
            title += " (Coeff.)"
        else:
            title += " (IR)"

    if is_coeff:
        b = convolve_coeffs(vb)
        a = convolve_coeffs(va)
        plot_from_coeff(b, a, n=2400, title=title, fs=args.rate,
                        show=args.show, save=args.save)
    else:
        plot_from_ir(ir, n=2400, title=title, fs=args.rate,
                     show=args.show, save=args.save)

    exit(0)


if __name__ == "__main__":
    main()
