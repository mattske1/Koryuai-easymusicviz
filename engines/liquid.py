#!/usr/bin/env python3
"""
visualizer.py — audio-reactive visualizer.

Two modes:
  loop        short seamlessly-tiling loop; the track's whole energy arc is
              folded into the loop phase (for Shorts/Reels promos).
  full-length the full song timeline rendered straight through — continuous
              flow, never repeats — with the song's audio muxed in
              (upload-ready video).

Usage:
    python3 visualizer.py in.wav --out out/loop.mp4 --seconds 10 --palette breakgeist
    python3 visualizer.py in.wav --out out/full.mp4 --full-length --palette breakgeist

Palettes live in PALETTES below; add one per band.
"""
import argparse, sys, math, os
import numpy as np
from multiprocessing import Pool

from common import (
    HAS_NUMBA, numba, _hex, palette_lut,
    analyze, fold_to_loop, resample_to_frames,
    make_grids, fbm, open_encoder, mux_audio,
)

# ---------------------------------------------------------------- palettes
# Stops are (position 0..1, (r,g,b) 0..255). Dark electronic default first.

PALETTES = {
    # Breakgeist: near-black indigo -> violet -> magenta -> ember white
    "breakgeist": [
        (0.00, _hex("#05050e")),
        (0.35, _hex("#12122e")),
        (0.58, _hex("#3b1d6e")),
        (0.76, _hex("#8a2be2")),
        (0.89, _hex("#ff2e88")),
        (1.00, _hex("#ffd9a0")),
    ],
    # frost: ice cyan ghost
    "frost": [
        (0.00, _hex("#04090d")),
        (0.35, _hex("#0b1e28")),
        (0.58, _hex("#155e75")),
        (0.76, _hex("#22d3ee")),
        (0.89, _hex("#a5f3fc")),
        (1.00, _hex("#ffffff")),
    ],
    # ember: red/orange heat
    "ember": [
        (0.00, _hex("#0d0505")),
        (0.35, _hex("#2b0f0a")),
        (0.58, _hex("#7c2d12")),
        (0.76, _hex("#ea580c")),
        (0.89, _hex("#fb923c")),
        (1.00, _hex("#fef3c7")),
    ],
}

# ---------------------------------------------------------------- fast path (numba)

if HAS_NUMBA:
    @numba.njit
    def _bilerp(grid, G, x, y):
        # x, y in [0, G). Mirrors common.sample_tileable.
        xi = int(np.floor(x)); yi = int(np.floor(y))
        fx = x - xi; fy = y - yi
        x0 = xi % G; y0 = yi % G
        x1 = (xi + 1) % G; y1 = (yi + 1) % G
        return (grid[y0, x0] * (1 - fx) * (1 - fy) +
                grid[y0, x1] * fx * (1 - fy) +
                grid[y1, x0] * (1 - fx) * fy +
                grid[y1, x1] * fx * fy)

    @numba.njit(parallel=True)
    def _field_fast(u0, v0, grids, wgrids, sizes,
                    ou1, ov1, ou2, ov2, du, dv, warp_amp, zoom, out):
        # Fused warp (2 octaves) + main field (4 octaves), one pass per pixel.
        h, w = u0.shape
        for j in numba.prange(h):
            for i in range(w):
                u = u0[j, i]; v = v0[j, i]
                wu = 0.0; wv = 0.0; asum = 0.0
                for oi in range(2):
                    amp = 0.5 ** oi
                    G = sizes[oi]; g = wgrids[oi]
                    wu += amp * _bilerp(g, G, ((u + ou1) % 1.0) * G,
                                             ((v + ov1) % 1.0) * G)
                    wv += amp * _bilerp(g, G, ((u + ou2) % 1.0) * G,
                                             ((v + ov2) % 1.0) * G)
                    asum += amp
                wu /= asum; wv /= asum
                su = (((u - 0.5) * zoom + 0.5 + du) * 3.0
                      + warp_amp * (wu - 0.5)) % 1.0
                sv = (((v - 0.5) * zoom + 0.5 + dv) * 3.0
                      + warp_amp * (wv - 0.5)) % 1.0
                val = 0.0; asum2 = 0.0
                for oi in range(4):
                    amp = 0.5 ** oi
                    G = sizes[oi]; g = grids[oi]
                    val += amp * _bilerp(g, G, su * G, sv * G)
                    asum2 += amp
                out[j, i] = val / asum2

def render_frame_fast(job):
    """Same look as render_frame; field computed by the fused numba kernel."""
    i, t, e, b, p, loop = job
    ctx = _CTX
    w, h, seed = ctx["w"], ctx["h"], ctx["seed"]
    u0, v0 = ctx["u0"], ctx["v0"]
    lut, vig = ctx["lut"], ctx["vig"]
    TAU = 2 * math.pi

    if loop:
        du = 0.30 * math.sin(TAU * t + 0.7) + 0.12 * math.sin(2 * TAU * t + 2.3)
        dv = 0.30 * math.cos(TAU * t + 2.1) + 0.12 * math.cos(2 * TAU * t + 0.4)
        ou1, ov1 = 0.5 * math.sin(TAU * t + 1.1), 0.5 * math.cos(TAU * t + 0.3)
        ou2, ov2 = 0.5 * math.sin(TAU * t + 2.6), 0.5 * math.cos(TAU * t + 1.9)
    else:
        du = 0.045 * t + 0.30 * math.sin(TAU * t / 41.0 + 0.7) \
             + 0.10 * math.sin(TAU * t / 13.0 + 2.3)
        dv = 0.036 * t + 0.30 * math.cos(TAU * t / 37.0 + 2.1) \
             + 0.10 * math.cos(TAU * t / 17.0 + 0.4)
        ou1, ov1 = 0.020 * t, 0.013 * t
        ou2, ov2 = 0.017 * t, 0.011 * t

    warp_amp = 0.35 + 2.2 * e
    zoom = 1.0 + 0.035 * p
    val = np.empty((h, w))
    _field_fast(u0, v0, ctx["fgrids"], ctx["fwgrids"], ctx["fsizes"],
                ou1, ov1, ou2, ov2, du, dv, warp_amp, zoom, val)

    # intensity sets where the noise field sits on the palette (same as slow path)
    contrast = 1.1 + 1.9 * e
    level = 0.5 - 0.52 * (1 - e)
    r = np.clip(level + (val - 0.5) * contrast + 0.10 * b * val, 0, 1)

    idx = np.clip((r * (len(lut) - 1)).astype(int), 0, len(lut) - 1)
    col = lut[idx]

    desat = (1 - e) * 0.55
    gray = col.mean(axis=2, keepdims=True) * np.array([0.75, 0.8, 1.0])
    col = col * (1 - desat) + gray * desat

    hot = np.clip((r - 0.72) / 0.28, 0, 1)[..., None]
    col *= (1 + 0.14 * p * hot)

    col *= vig[..., None]

    grng = np.random.default_rng(seed * 100003 + i)
    grain = grng.standard_normal((h, w, 1)) * (0.010 + 0.018 * e)
    col = np.clip(col + grain, 0, 1)

    return (col * 255).astype(np.uint8).tobytes()

# ---------------------------------------------------------------- frames

def make_ctx(w, h, seed, palette_name):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    u0 = xx / w
    v0 = yy / h
    cx, cy = (xx / w - 0.5) * 2, (yy / h - 0.5) * 2
    vig = 1.0 - 0.32 * np.clip((cx**2 + cy**2) / 2.0, 0, 1) ** 1.2
    ctx = {"w": w, "h": h, "seed": seed,
           "grids": make_grids(seed),
           "warp_grids": make_grids(seed + 999),
           "lut": palette_lut(PALETTES[palette_name]),
           "u0": u0, "v0": v0, "vig": vig}
    if HAS_NUMBA:
        # numba-friendly: tuples of arrays + sizes
        ctx["fgrids"] = tuple(g for g, _ in ctx["grids"])
        ctx["fwgrids"] = tuple(g for g, _ in ctx["warp_grids"])
        ctx["fsizes"] = (8, 16, 32, 64)
    return ctx

_CTX = None

def _init_worker(ctx):
    global _CTX
    _CTX = ctx

def render_frame(job):
    """job = (i, t, e, b, p, loop). t = loop phase [0,1) or seconds."""
    i, t, e, b, p, loop = job
    ctx = _CTX
    w, h, seed = ctx["w"], ctx["h"], ctx["seed"]
    u0, v0 = ctx["u0"], ctx["v0"]
    grids, warp_grids = ctx["grids"], ctx["warp_grids"]
    lut, vig = ctx["lut"], ctx["vig"]
    TAU = 2 * math.pi

    if loop:
        # periodic drift path — tiles exactly
        du = 0.30 * math.sin(TAU * t + 0.7) + 0.12 * math.sin(2 * TAU * t + 2.3)
        dv = 0.30 * math.cos(TAU * t + 2.1) + 0.12 * math.cos(2 * TAU * t + 0.4)
        wu = fbm((u0 + 0.5 * math.sin(TAU * t + 1.1)) % 1.0,
                 (v0 + 0.5 * math.cos(TAU * t + 0.3)) % 1.0, warp_grids, 2)
        wv = fbm((v0 + 0.5 * math.sin(TAU * t + 2.6)) % 1.0,
                 (u0 + 0.5 * math.cos(TAU * t + 1.9)) % 1.0, warp_grids, 2)
    else:
        # continuous flow — never repeats
        du = 0.045 * t + 0.30 * math.sin(TAU * t / 41.0 + 0.7) \
             + 0.10 * math.sin(TAU * t / 13.0 + 2.3)
        dv = 0.036 * t + 0.30 * math.cos(TAU * t / 37.0 + 2.1) \
             + 0.10 * math.cos(TAU * t / 17.0 + 0.4)
        wu = fbm((u0 + 0.020 * t) % 1.0, (v0 + 0.013 * t) % 1.0, warp_grids, 2)
        wv = fbm((v0 + 0.017 * t) % 1.0, (u0 + 0.011 * t) % 1.0, warp_grids, 2)

    warp_amp = 0.35 + 2.2 * e
    zoom = 1.0 + 0.035 * p
    su = (((u0 - 0.5) * zoom + 0.5 + du) * 3.0 + warp_amp * (wu - 0.5)) % 1.0
    sv = (((v0 - 0.5) * zoom + 0.5 + dv) * 3.0 + warp_amp * (wv - 0.5)) % 1.0
    val = fbm(su, sv, grids, 4)

    # intensity sets where the noise field sits on the palette:
    # quiet -> deep darks with faint texture; loud -> hot filaments
    contrast = 1.1 + 1.9 * e
    level = 0.5 - 0.52 * (1 - e)
    r = np.clip(level + (val - 0.5) * contrast + 0.10 * b * val, 0, 1)

    idx = np.clip((r * (len(lut) - 1)).astype(int), 0, len(lut) - 1)
    col = lut[idx]  # h,w,3

    # low energy desaturates toward deep neutral
    desat = (1 - e) * 0.55
    gray = col.mean(axis=2, keepdims=True) * np.array([0.75, 0.8, 1.0])
    col = col * (1 - desat) + gray * desat

    # beat swell on the hot regions
    hot = np.clip((r - 0.72) / 0.28, 0, 1)[..., None]
    col *= (1 + 0.14 * p * hot)

    col *= vig[..., None]

    # grain, seeded per frame index
    grng = np.random.default_rng(seed * 100003 + i)
    grain = grng.standard_normal((h, w, 1)) * (0.010 + 0.018 * e)
    col = np.clip(col + grain, 0, 1)

    return (col * 255).astype(np.uint8).tobytes()

# ---------------------------------------------------------------- encode

def render_loop(analysis, palette_name, seconds=10, fps=30, w=540, h=960,
                seed=7, out_path="out/loop.mp4", upscale=(1080, 1920), crf=20,
                fast=False, preset="medium", scaler="lanczos"):
    global _CTX
    dur = analysis["duration"]
    n_frames = int(seconds * fps)
    I = fold_to_loop(analysis["intensity"], dur, seconds, n_frames)
    B = fold_to_loop(analysis["brightness"], dur, seconds, n_frames)
    P = fold_to_loop(analysis["pulse"], dur, seconds, n_frames)
    _CTX = make_ctx(w, h, seed, palette_name)
    if fast and not HAS_NUMBA:
        print("  WARNING: numba not available, using slow path", flush=True)
        fast = False
    frame_fn = render_frame_fast if fast else render_frame
    if fast:
        print("  rendering with numba fast path...", flush=True)
    proc = open_encoder(w, h, fps, out_path, crf, upscale, preset, scaler)
    for i in range(n_frames):
        t = i / n_frames
        proc.stdin.write(frame_fn((i, t, I[i], B[i], P[i], True)))
        if (i + 1) % 60 == 0:
            print(f"  frame {i+1}/{n_frames}", flush=True)
    proc.stdin.close()
    proc.wait()
    return out_path

def render_full(analysis, audio_src, palette_name, fps=30, w=540, h=960,
                seed=7, out_path="out/full.mp4", upscale=(1080, 1920), crf=20,
                workers=None, fast=False, preset="medium", scaler="lanczos"):
    """Full song length, continuous variation, song audio muxed in."""
    global _CTX
    dur = analysis["duration"]
    n_frames = int(dur * fps)
    print(f"  full-length: {dur:.1f}s -> {n_frames} frames", flush=True)
    I = resample_to_frames(analysis["intensity"], dur, n_frames, fps)
    B = resample_to_frames(analysis["brightness"], dur, n_frames, fps)
    P = resample_to_frames(analysis["pulse"], dur, n_frames, fps)
    ctx = make_ctx(w, h, seed, palette_name)
    if fast and not HAS_NUMBA:
        print("  WARNING: numba not available, using slow path", flush=True)
        fast = False
    tmp = out_path + ".silent.mp4"
    jobs = [(i, i / fps, I[i], B[i], P[i], False) for i in range(n_frames)]
    if fast:
        # numba parallel=True saturates all cores in one process; the
        # multiprocess pool would only add IPC overhead.
        print("  rendering with numba fast path...", flush=True)
        _CTX = ctx
        proc = open_encoder(w, h, fps, tmp, crf, upscale, preset, scaler)
        done = 0
        for job in jobs:
            proc.stdin.write(render_frame_fast(job))
            done += 1
            if done % 300 == 0:
                print(f"  frame {done}/{n_frames}", flush=True)
        proc.stdin.close()
        proc.wait()
    else:
        if workers is None:
            workers = max(1, os.cpu_count() or 1)
        print(f"  rendering with {workers} workers...", flush=True)
        with Pool(workers, initializer=_init_worker, initargs=(ctx,)) as pool:
            proc = open_encoder(w, h, fps, tmp, crf, upscale, preset, scaler)
            done = 0
            for buf in pool.imap(render_frame, jobs, chunksize=8):
                proc.stdin.write(buf)
                done += 1
                if done % 300 == 0:
                    print(f"  frame {done}/{n_frames}", flush=True)
            proc.stdin.close()
            proc.wait()
    print("  muxing audio...", flush=True)
    mux_audio(tmp, audio_src, out_path)
    return out_path

# ---------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="audio file (wav/mp3/...)")
    ap.add_argument("--out", default="out/loop.mp4")
    ap.add_argument("--seconds", type=float, default=10,
                    help="loop length (loop mode only)")
    ap.add_argument("--full-length", action="store_true",
                    help="render the whole song, continuous variation + audio muxed")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=540)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--palette", default="breakgeist", choices=list(PALETTES))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--no-upscale", action="store_true")
    ap.add_argument("--upscale-factor", type=int, default=2,
                    help="upscale multiplier (2=standard, 4=turbo)")
    ap.add_argument("--fast", action="store_true",
                    help="numba fast path (fused noise kernel, uses all cores)")
    ap.add_argument("--x264-preset", default="medium",
                    help="x264 preset (medium=quality, veryfast=speed)")
    ap.add_argument("--scaler", default="lanczos",
                    help="ffmpeg upscale scaler (lanczos=quality, bilinear=speed)")
    args = ap.parse_args()

    print("analyzing audio...", flush=True)
    an = analyze(args.input)
    print(f"  duration {an['duration']:.1f}s  "
          f"mean intensity {an['intensity'].mean():.2f}", flush=True)
    upscale = None if args.no_upscale else (args.width * args.upscale_factor,
                                                 args.height * args.upscale_factor)
    if args.full_length:
        print("rendering full-length...", flush=True)
        out = render_full(an, args.input, args.palette, fps=args.fps,
                          w=args.width, h=args.height, seed=args.seed,
                          out_path=args.out, upscale=upscale, crf=args.crf,
                          workers=args.workers, fast=args.fast,
                          preset=args.x264_preset, scaler=args.scaler)
    else:
        print("rendering loop...", flush=True)
        out = render_loop(an, args.palette, seconds=args.seconds, fps=args.fps,
                          w=args.width, h=args.height, seed=args.seed,
                          out_path=args.out, upscale=upscale, crf=args.crf,
                          fast=args.fast, preset=args.x264_preset,
                          scaler=args.scaler)
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
