#!/usr/bin/env python3
"""galaxy.py — audio-reactive galaxy / deep-space visualizer.

Programmatic engine (numpy/numba, CPU-only, no model calls): a spiral
galaxy with domain-warped nebula clouds, a twinkling parallax starfield,
and a pulse-flaring core — all driven by the song's intensity/brightness/
pulse envelopes.

Two modes (same as liquid):
  loop        seamlessly-tiling loop; envelopes folded into the loop phase.
              The arms rotate exactly half a turn per loop (2-fold symmetry
              -> seamless) and star twinkle uses integer cycles.
  full-length the full song straight through, continuous drift + song audio
              muxed in.

Usage:
    python3 galaxy.py in.wav --out out/loop.mp4 --seconds 10 --palette nebula
    python3 galaxy.py in.wav --out out/full.mp4 --full-length --palette quasar
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

PALETTES = {
    # nebula: deep indigo -> violet -> cyan-white
    "nebula": [
        (0.00, _hex("#020208")),
        (0.35, _hex("#0d0b26")),
        (0.58, _hex("#3b1d6e")),
        (0.76, _hex("#8a2be2")),
        (0.89, _hex("#22d3ee")),
        (1.00, _hex("#e8fbff")),
    ],
    # quasar: amber / gold galaxy
    "quasar": [
        (0.00, _hex("#050403")),
        (0.35, _hex("#1a0f08")),
        (0.58, _hex("#7c2d12")),
        (0.76, _hex("#f59e0b")),
        (0.89, _hex("#fde68a")),
        (1.00, _hex("#ffffff")),
    ],
    # void: magenta / purple deep field
    "void": [
        (0.00, _hex("#020204")),
        (0.35, _hex("#0a0a18")),
        (0.58, _hex("#2d1b4e")),
        (0.76, _hex("#7c3aed")),
        (0.89, _hex("#f0abfc")),
        (1.00, _hex("#ffffff")),
    ],
}

TAU = 2 * math.pi

# ---------------------------------------------------------------- fast path (numba)

if HAS_NUMBA:
    @numba.njit
    def _bilerp(grid, G, x, y):
        xi = int(np.floor(x)); yi = int(np.floor(y))
        fx = x - xi; fy = y - yi
        x0 = xi % G; y0 = yi % G
        x1 = (xi + 1) % G; y1 = (yi + 1) % G
        return (grid[y0, x0] * (1 - fx) * (1 - fy) +
                grid[y0, x1] * fx * (1 - fy) +
                grid[y1, x0] * (1 - fx) * fy +
                grid[y1, x1] * fx * fy)

    @numba.njit(parallel=True)
    def _field_fast(u0, v0, aspect, grids, wgrids, sizes,
                    ou1, ov1, ou2, ov2, od1, od2, du, dv,
                    warp_amp, rot, e, b, p, zoom, neb_amt, lut, out):
        # Per pixel: warped nebula fbm + logarithmic spiral arms with dust
        # lanes + flaring core -> palette-mapped RGB in out (h, w, 3).
        h, w = u0.shape
        neb_lvl = 0.30 + 0.70 * e
        arm_lvl = 0.35 + 0.65 * e
        core_gain = 1.0 + 2.2 * p
        nl = len(lut)
        for j in numba.prange(h):
            for i in range(w):
                uu = (u0[j, i] - 0.5) * zoom + 0.5
                vv = (v0[j, i] - 0.5) * zoom + 0.5
                # domain warp (2 octaves)
                wu = 0.0; wv = 0.0; asum = 0.0
                for oi in range(2):
                    amp = 0.5 ** oi
                    G = sizes[oi]; g = wgrids[oi]
                    wu += amp * _bilerp(g, G, ((uu + ou1) % 1.0) * G,
                                             ((vv + ov1) % 1.0) * G)
                    wv += amp * _bilerp(g, G, ((uu + ou2) % 1.0) * G,
                                             ((vv + ov2) % 1.0) * G)
                    asum += amp
                wu /= asum; wv /= asum
                # nebula (4 octaves)
                su = (uu * 3.0 + du + warp_amp * (wu - 0.5)) % 1.0
                sv = (vv * 3.0 + dv + warp_amp * (wv - 0.5)) % 1.0
                neb = 0.0; asum2 = 0.0
                for oi in range(4):
                    amp = 0.5 ** oi
                    G = sizes[oi]; g = grids[oi]
                    neb += amp * _bilerp(g, G, su * G, sv * G)
                    asum2 += amp
                neb /= asum2
                # dust lanes (2 octaves, higher frequency)
                su2 = (uu * 6.0 + od1) % 1.0
                sv2 = (vv * 6.0 + od2) % 1.0
                dust = 0.0
                for oi in range(2):
                    amp = 0.5 ** oi
                    G = sizes[oi]; g = grids[oi]
                    dust += amp * _bilerp(g, G, su2 * G, sv2 * G)
                dust /= 1.5
                # spiral arms: logarithmic, 2-fold
                dx = (uu - 0.5) * aspect
                dy = vv - 0.5
                rr = np.sqrt(dx * dx + dy * dy)
                th = np.arctan2(dy, dx)
                arm_ph = 2.0 * (th - rot) - 5.0 * np.log(rr + 1e-3)
                c = 0.5 + 0.5 * np.cos(arm_ph)
                arms = c * c * c * np.exp(-rr / 0.30)
                arms *= 0.55 + 0.45 * dust
                # core flare
                cr = rr / 0.055
                core = np.exp(-cr * cr) * core_gain
                val = (neb * 0.30 * neb_lvl * neb_amt
                       + arms * 0.72 * arm_lvl
                       + core * 0.95)
                r = val + 0.08 * b * val
                if r < 0.0:
                    r = 0.0
                elif r > 1.0:
                    r = 1.0
                idx = int(r * (nl - 1))
                out[j, i, 0] = lut[idx, 0]
                out[j, i, 1] = lut[idx, 1]
                out[j, i, 2] = lut[idx, 2]

    @numba.njit
    def _splat_stars(img, xs, ys, sigmas, bases, tints,
                     tw_f, tw_p, vx, vy, T, TAU_, star_gain):
        # Serial over stars: additive gaussian splats + hero cross streaks.
        h, w, _ = img.shape
        n = xs.shape[0]
        for s in range(n):
            tw = 0.55 + 0.45 * np.sin(TAU_ * (tw_f[s] * T + tw_p[s]))
            bright = bases[s] * tw * star_gain
            if bright < 0.004:
                continue
            x = (xs[s] + vx[s] * T) % 1.0 * w
            y = (ys[s] + vy[s] * T) % 1.0 * h
            sig = sigmas[s]
            R = int(3.0 * sig) + 1
            cx = int(round(x)); cy = int(round(y))
            hero = bases[s] > 0.90
            for dyy in range(-R, R + 1):
                jy = cy + dyy
                if jy < 0 or jy >= h:
                    continue
                for dxx in range(-R, R + 1):
                    ix = cx + dxx
                    if ix < 0 or ix >= w:
                        continue
                    d2 = float(dxx * dxx + dyy * dyy)
                    g = np.exp(-d2 / (2.0 * sig * sig))
                    v = bright * g
                    if hero:
                        # cross streaks on the brightest stars
                        streak = (np.exp(-(dyy * dyy) / 0.5) *
                                  np.exp(-abs(float(dxx)) / 7.0) +
                                  np.exp(-(dxx * dxx) / 0.5) *
                                  np.exp(-abs(float(dyy)) / 7.0))
                        v += bright * 0.55 * streak
                    img[jy, ix, 0] += tints[s, 0] * v
                    img[jy, ix, 1] += tints[s, 1] * v
                    img[jy, ix, 2] += tints[s, 2] * v


def render_frame_fast(job):
    """job = (i, t, e, b, p, loop). t = loop phase [0,1) or seconds."""
    i, t, e, b, p, loop = job
    ctx = _CTX
    w, h, seed = ctx["w"], ctx["h"], ctx["seed"]
    u0, v0 = ctx["u0"], ctx["v0"]
    lut, vig = ctx["lut"], ctx["vig"]

    if loop:
        # periodic paths — everything tiles exactly
        rot = math.pi * t                      # half turn; m=2 -> seamless
        du = 0.25 * math.sin(TAU * t + 0.9)
        dv = 0.25 * math.cos(TAU * t + 2.2)
        ou1, ov1 = 0.4 * math.sin(TAU * t + 1.1), 0.4 * math.cos(TAU * t + 0.3)
        ou2, ov2 = 0.4 * math.sin(TAU * t + 2.6), 0.4 * math.cos(TAU * t + 1.9)
        od1, od2 = 0.3 * math.sin(TAU * t + 0.5), 0.3 * math.cos(TAU * t + 1.4)
        T = t
    else:
        rot = 0.06 * t
        du = 0.020 * t + 0.25 * math.sin(TAU * t / 47.0 + 0.9)
        dv = 0.016 * t + 0.25 * math.cos(TAU * t / 53.0 + 2.2)
        ou1, ov1 = 0.015 * t, 0.011 * t
        ou2, ov2 = 0.012 * t, 0.009 * t
        od1, od2 = 0.010 * t, 0.008 * t
        T = t

    warp_amp = 0.5 + 1.6 * e
    zoom = 1.0 + 0.025 * p
    col = np.empty((h, w, 3))
    _field_fast(u0, v0, ctx["aspect"], ctx["fgrids"], ctx["fwgrids"],
                ctx["fsizes"], ou1, ov1, ou2, ov2, od1, od2, du, dv,
                warp_amp, rot, e, b, p, zoom, ctx["neb_amt"], lut, col)

    star_gain = 0.45 + 0.55 * e
    _splat_stars(col, ctx["sx"], ctx["sy"], ctx["ssig"], ctx["sbase"],
                 ctx["stint"], ctx["stw_f"], ctx["stw_p"],
                 ctx["svx"], ctx["svy"], T, TAU, star_gain)

    col *= vig[..., None]

    grng = np.random.default_rng(seed * 100003 + i)
    grain = grng.standard_normal((h, w, 1)) * (0.010 + 0.018 * e)
    col = np.clip(col + grain, 0, 1)

    return (col * 255).astype(np.uint8).tobytes()

# ---------------------------------------------------------------- frames (numpy slow path)

def render_frame(job):
    i, t, e, b, p, loop = job
    ctx = _CTX
    w, h, seed = ctx["w"], ctx["h"], ctx["seed"]
    u0, v0 = ctx["u0"], ctx["v0"]
    grids, warp_grids = ctx["grids"], ctx["warp_grids"]
    lut, vig = ctx["lut"], ctx["vig"]
    aspect = ctx["aspect"]

    if loop:
        rot = math.pi * t
        du = 0.25 * math.sin(TAU * t + 0.9)
        dv = 0.25 * math.cos(TAU * t + 2.2)
        wu = fbm((u0 + 0.4 * math.sin(TAU * t + 1.1)) % 1.0,
                 (v0 + 0.4 * math.cos(TAU * t + 0.3)) % 1.0, warp_grids, 2)
        wv = fbm((u0 + 0.4 * math.sin(TAU * t + 2.6)) % 1.0,
                 (v0 + 0.4 * math.cos(TAU * t + 1.9)) % 1.0, warp_grids, 2)
        dust = fbm((u0 * 6.0 + 0.3 * math.sin(TAU * t + 0.5)) % 1.0,
                   (v0 * 6.0 + 0.3 * math.cos(TAU * t + 1.4)) % 1.0, grids, 2)
        T = t
    else:
        rot = 0.06 * t
        du = 0.020 * t + 0.25 * math.sin(TAU * t / 47.0 + 0.9)
        dv = 0.016 * t + 0.25 * math.cos(TAU * t / 53.0 + 2.2)
        wu = fbm((u0 + 0.015 * t) % 1.0, (v0 + 0.011 * t) % 1.0,
                 warp_grids, 2)
        wv = fbm((u0 + 0.012 * t) % 1.0, (v0 + 0.009 * t) % 1.0,
                 warp_grids, 2)
        dust = fbm((u0 * 6.0 + 0.010 * t) % 1.0,
                   (v0 * 6.0 + 0.008 * t) % 1.0, grids, 2)
        T = t

    warp_amp = 0.5 + 1.6 * e
    zoom = 1.0 + 0.025 * p
    uu = (u0 - 0.5) * zoom + 0.5
    vv = (v0 - 0.5) * zoom + 0.5
    su = (uu * 3.0 + du + warp_amp * (wu - 0.5)) % 1.0
    sv = (vv * 3.0 + dv + warp_amp * (wv - 0.5)) % 1.0
    neb = fbm(su, sv, grids, 4)

    dx = (uu - 0.5) * aspect
    dy = vv - 0.5
    rr = np.sqrt(dx * dx + dy * dy)
    th = np.arctan2(dy, dx)
    arm_ph = 2.0 * (th - rot) - 5.0 * np.log(rr + 1e-3)
    c = 0.5 + 0.5 * np.cos(arm_ph)
    arms = c ** 3 * np.exp(-rr / 0.30) * (0.55 + 0.45 * dust)
    core = np.exp(-(rr / 0.055) ** 2) * (1.0 + 2.2 * p)

    neb_lvl = 0.30 + 0.70 * e
    arm_lvl = 0.35 + 0.65 * e
    val = (neb * 0.30 * neb_lvl * ctx["neb_amt"]
           + arms * 0.72 * arm_lvl + core * 0.95)
    r = np.clip(val + 0.08 * b * val, 0, 1)
    idx = np.clip((r * (len(lut) - 1)).astype(int), 0, len(lut) - 1)
    col = lut[idx].copy()

    # stars (numpy slow path)
    tw = 0.55 + 0.45 * np.sin(TAU * (ctx["stw_f"] * T + ctx["stw_p"]))
    bright = ctx["sbase"] * tw * (0.45 + 0.55 * e)
    xs = ((ctx["sx"] + ctx["svx"] * T) % 1.0 * w).astype(int)
    ys = ((ctx["sy"] + ctx["svy"] * T) % 1.0 * h).astype(int)
    for k in range(len(xs)):
        if bright[k] < 0.004:
            continue
        sig = ctx["ssig"][k]
        R = int(3.0 * sig) + 1
        x0p, y0p = xs[k] - R, ys[k] - R
        x1p, y1p = xs[k] + R + 1, ys[k] + R + 1
        if x1p < 0 or y1p < 0 or x0p >= w or y0p >= h:
            continue
        ax0, ay0 = max(0, x0p), max(0, y0p)
        ax1, ay1 = min(w, x1p), min(h, y1p)
        dyy, dxx = np.mgrid[ay0 - ys[k]:ay1 - ys[k], ax0 - xs[k]:ax1 - xs[k]]
        g = np.exp(-(dxx ** 2 + dyy ** 2) / (2.0 * sig * sig))
        v = bright[k] * g
        if ctx["sbase"][k] > 0.90:
            streak = (np.exp(-(dyy ** 2) / 0.5) * np.exp(-np.abs(dxx) / 7.0)
                      + np.exp(-(dxx ** 2) / 0.5) * np.exp(-np.abs(dyy) / 7.0))
            v = v + bright[k] * 0.55 * streak
        col[ay0:ay1, ax0:ax1] += (ctx["stint"][k] * v[..., None])

    col *= vig[..., None]
    grng = np.random.default_rng(seed * 100003 + i)
    grain = grng.standard_normal((h, w, 1)) * (0.010 + 0.018 * e)
    col = np.clip(col + grain, 0, 1)
    return (col * 255).astype(np.uint8).tobytes()

# ---------------------------------------------------------------- ctx

def make_ctx(w, h, seed, palette_name, n_stars=2200, neb_amt=1.0, loop=True):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    u0 = xx / w
    v0 = yy / h
    cx, cy = (xx / w - 0.5) * 2, (yy / h - 0.5) * 2
    vig = 1.0 - 0.32 * np.clip((cx**2 + cy**2) / 2.0, 0, 1) ** 1.2

    rng = np.random.default_rng(seed + 4242)
    sx = rng.random(n_stars)
    sy = rng.random(n_stars)
    ssig = 0.6 + 1.6 * rng.random(n_stars) ** 2
    sbase = 0.15 + 0.85 * rng.random(n_stars) ** 3
    if loop:
        stw_f = rng.integers(1, 4, n_stars).astype(np.float64)  # int cycles
        svx = rng.uniform(-0.02, 0.02, n_stars)   # per loop phase
        svy = rng.uniform(-0.02, 0.02, n_stars)
    else:
        stw_f = rng.uniform(0.4, 1.6, n_stars)   # Hz
        svx = rng.uniform(-0.003, 0.003, n_stars)  # per second
        svy = rng.uniform(-0.003, 0.003, n_stars)
    stw_p = rng.random(n_stars)
    # star tints: cool blue-white -> neutral -> warm
    temp = rng.random(n_stars)
    stint = np.stack([
        0.75 + 0.25 * temp,
        0.82 + 0.14 * (1 - np.abs(temp - 0.5) * 2),
        1.00 - 0.22 * temp,
    ], axis=1)

    ctx = {"w": w, "h": h, "seed": seed, "aspect": w / h,
           "grids": make_grids(seed),
           "warp_grids": make_grids(seed + 999),
           "lut": palette_lut(PALETTES[palette_name]),
           "u0": u0, "v0": v0, "vig": vig,
           "neb_amt": neb_amt,
           "sx": sx, "sy": sy, "ssig": ssig, "sbase": sbase,
           "stint": stint, "stw_f": stw_f, "stw_p": stw_p,
           "svx": svx, "svy": svy}
    if HAS_NUMBA:
        ctx["fgrids"] = tuple(g for g, _ in ctx["grids"])
        ctx["fwgrids"] = tuple(g for g, _ in ctx["warp_grids"])
        ctx["fsizes"] = (8, 16, 32, 64)
    return ctx

_CTX = None

def _init_worker(ctx):
    global _CTX
    _CTX = ctx

# ---------------------------------------------------------------- encode

def render_loop(analysis, palette_name, seconds=10, fps=30, w=540, h=960,
                seed=7, out_path="out/loop.mp4", upscale=(1080, 1920), crf=20,
                fast=False, preset="medium", scaler="lanczos",
                n_stars=2200, neb_amt=1.0):
    global _CTX
    dur = analysis["duration"]
    n_frames = int(seconds * fps)
    I = fold_to_loop(analysis["intensity"], dur, seconds, n_frames)
    B = fold_to_loop(analysis["brightness"], dur, seconds, n_frames)
    P = fold_to_loop(analysis["pulse"], dur, seconds, n_frames)
    _CTX = make_ctx(w, h, seed, palette_name, n_stars, neb_amt, loop=True)
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
                workers=None, fast=False, preset="medium", scaler="lanczos",
                n_stars=2200, neb_amt=1.0):
    """Full song length, continuous variation, song audio muxed in."""
    global _CTX
    dur = analysis["duration"]
    n_frames = int(dur * fps)
    print(f"  full-length: {dur:.1f}s -> {n_frames} frames", flush=True)
    I = resample_to_frames(analysis["intensity"], dur, n_frames, fps)
    B = resample_to_frames(analysis["brightness"], dur, n_frames, fps)
    P = resample_to_frames(analysis["pulse"], dur, n_frames, fps)
    ctx = make_ctx(w, h, seed, palette_name, n_stars, neb_amt, loop=False)
    if fast and not HAS_NUMBA:
        print("  WARNING: numba not available, using slow path", flush=True)
        fast = False
    tmp = out_path + ".silent.mp4"
    jobs = [(i, i / fps, I[i], B[i], P[i], False) for i in range(n_frames)]
    if fast:
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
    ap.add_argument("--palette", default="nebula", choices=list(PALETTES))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--stars", type=int, default=2200,
                    help="star count in the starfield")
    ap.add_argument("--nebula", type=float, default=1.0,
                    help="nebula cloud amount (0..1.5)")
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--no-upscale", action="store_true")
    ap.add_argument("--upscale-factor", type=int, default=2,
                    help="upscale multiplier (2=standard, 4=turbo)")
    ap.add_argument("--fast", action="store_true",
                    help="numba fast path (fused kernels, uses all cores)")
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
    kw = dict(n_stars=args.stars, neb_amt=args.nebula)
    if args.full_length:
        print("rendering full-length...", flush=True)
        out = render_full(an, args.input, args.palette, fps=args.fps,
                          w=args.width, h=args.height, seed=args.seed,
                          out_path=args.out, upscale=upscale, crf=args.crf,
                          workers=args.workers, fast=args.fast,
                          preset=args.x264_preset, scaler=args.scaler, **kw)
    else:
        print("rendering loop...", flush=True)
        out = render_loop(an, args.palette, seconds=args.seconds, fps=args.fps,
                          w=args.width, h=args.height, seed=args.seed,
                          out_path=args.out, upscale=upscale, crf=args.crf,
                          fast=args.fast, preset=args.x264_preset,
                          scaler=args.scaler, **kw)
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
