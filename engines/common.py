#!/usr/bin/env python3
"""common.py — shared helpers for koryuai-easymusicviz engines.

Audio analysis (intensity / brightness / pulse envelopes), envelope
folding for seamless loops, palette LUTs, tileable value-noise fbm,
and the ffmpeg rawvideo pipe encoder. Engines import what they need;
each engine file still runs standalone as a script.
"""
import math
import os
import subprocess

import numpy as np

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    numba = None
    HAS_NUMBA = False


# ---------------------------------------------------------------- palettes

def _hex(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def palette_lut(stops, n=1024):
    pos = np.array([s for s, _ in stops])
    cols = np.array([c for _, c in stops], dtype=np.float64) / 255.0
    xs = np.linspace(0, 1, n)
    return np.stack([np.interp(xs, pos, cols[:, ch]) for ch in range(3)],
                    axis=1)


# ---------------------------------------------------------------- analysis

def load_mono(path, sr=22050):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr),
           "-f", "f32le", "-acodec", "pcm_f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32), sr


def smooth(x, sigma_s, sr_env):
    sigma = max(1, int(sigma_s * sr_env))
    k = np.arange(-3 * sigma, 3 * sigma + 1)
    ker = np.exp(-0.5 * (k / sigma) ** 2)
    ker /= ker.sum()
    return np.convolve(x, ker, mode="same")


def norm_robust(x, lo=2, hi=98):
    a, b = np.percentile(x, [lo, hi])
    if b - a < 1e-9:
        return np.zeros_like(x)
    return np.clip((x - a) / (b - a), 0, 1)


def analyze(path):
    """Returns dict of envelopes sampled at env_sr Hz: intensity, brightness, pulse."""
    y, sr = load_mono(path)
    dur = len(y) / sr
    env_sr = 10.0  # envelope samples per second
    win = int(sr / env_sr)

    # --- intensity: RMS energy
    sq = y[: len(y) // win * win].reshape(-1, win)
    rms = np.sqrt((sq ** 2).mean(axis=1) + 1e-12)
    intensity = smooth(norm_robust(np.log1p(rms * 40)), 0.6, env_sr)

    # --- brightness: spectral centroid
    n_fft, hop = 2048, 1024
    frames = []
    for start in range(0, len(y) - n_fft, hop):
        seg = y[start:start + n_fft] * np.hanning(n_fft)
        mag = np.abs(np.fft.rfft(seg))
        f = np.fft.rfftfreq(n_fft, 1 / sr)
        c = (f * mag).sum() / (mag.sum() + 1e-9)
        frames.append(c)
    frames = np.array(frames)
    t_frames = np.arange(len(frames)) * hop / sr
    t_env = np.arange(len(rms)) / env_sr
    bright = np.interp(t_env, t_frames, frames)
    brightness = smooth(norm_robust(bright, 5, 95), 0.8, env_sr)

    # --- pulse: onset strength from log-energy deltas
    loge = np.log1p(rms * 40)
    onset = np.maximum(0, np.diff(loge, prepend=loge[0]))
    pulse = smooth(norm_robust(onset, 50, 99.5), 0.15, env_sr)

    return {"intensity": intensity, "brightness": brightness, "pulse": pulse,
            "env_sr": env_sr, "duration": dur, "sr": sr}


def fold_to_loop(env, duration, loop_s, n_frames):
    """Average the track envelope into one periodic loop phase."""
    t_env = np.arange(len(env)) / 10.0
    out = np.zeros(n_frames)
    for i in range(n_frames):
        base = (i / n_frames) * loop_s
        ks = np.arange(0, math.ceil(duration / loop_s))
        ts = base + ks * loop_s
        ts = ts[ts < duration - 0.05]
        if len(ts) == 0:
            continue
        out[i] = np.interp(ts, t_env, env).mean()
    # circular smoothing so the seam is invisible
    k = 9
    ker = np.exp(-0.5 * (np.arange(-k, k + 1) / 3.0) ** 2)
    ker /= ker.sum()
    padded = np.concatenate([out[-2 * k:], out, out[:2 * k]])
    return np.convolve(padded, ker, mode="same")[2 * k:2 * k + n_frames]


def resample_to_frames(env, duration, n_frames, fps):
    """Straight-through envelope at frame rate (full-length mode)."""
    t_env = np.arange(len(env)) / 10.0
    t_f = np.arange(n_frames) / fps
    out = np.interp(t_f, t_env, env)
    return smooth(out, 0.15, fps)


# ---------------------------------------------------------------- noise

def make_grids(seed, sizes=(8, 16, 32, 64)):
    rng = np.random.default_rng(seed)
    return [(rng.random((g, g)).astype(np.float64), g) for g in sizes]


def sample_tileable(grid, xs, ys):
    """Bilinear sample of a tileable grid. xs/ys in grid units."""
    G = grid.shape[0]
    xi = np.floor(xs)
    yi = np.floor(ys)
    x0 = (xi.astype(np.int64)) % G
    y0 = (yi.astype(np.int64)) % G
    x1 = (x0 + 1) % G
    y1 = (y0 + 1) % G
    fx = xs - xi
    fy = ys - yi
    fx = np.clip(fx, 0, 1)
    fy = np.clip(fy, 0, 1)
    return (grid[y0, x0] * (1 - fx) * (1 - fy) +
            grid[y0, x1] * fx * (1 - fy) +
            grid[y1, x0] * (1 - fx) * fy +
            grid[y1, x1] * fx * fy)


def fbm(u, v, grids, octaves=4):
    """u,v in [0,1). Returns field in ~[0,1]."""
    total = np.zeros_like(u)
    amp_sum = 0.0
    for (grid, G), oi in zip(grids, range(octaves)):
        amp = 0.5 ** oi
        total += amp * sample_tileable(grid, u * G, v * G)
        amp_sum += amp
    return total / amp_sum


# ---------------------------------------------------------------- encode

def open_encoder(w, h, fps, out_path, crf, upscale, preset="medium",
                 scaler="lanczos"):
    scale_args = []
    if upscale:
        scale_args = ["-vf", f"scale={upscale[0]}:{upscale[1]}:flags={scaler}"]
    cmd = ["ffmpeg", "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-",
           "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart",
           "-crf", str(crf), "-preset", preset] + scale_args + [out_path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def mux_audio(silent_path, audio_src, out_path):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", silent_path,
                    "-i", audio_src, "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "192k", "-movflags", "+faststart",
                    "-shortest", out_path], check=True)
    os.remove(silent_path)
