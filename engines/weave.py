#!/usr/bin/env python3
"""Weave a short clip into a seamless ambient loop of target duration.

Chains N copies of the clip with d-second crossfades, then crossfades the
tail back into the head so the loop point is invisible:
    total = N*L - N*d  =>  d = (N*L - target) / N
"""
import argparse, math, subprocess, sys


def probe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1", path],
        capture_output=True, text=True, check=True).stdout
    return float(out.strip().split("=")[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--target", type=float, required=True, help="loop seconds")
    ap.add_argument("--out", required=True)
    ap.add_argument("--xfade", type=float, default=None,
                    help="crossfade seconds (auto-solved if omitted)")
    args = ap.parse_args()

    L = probe_duration(args.clip)
    if args.xfade:
        d = args.xfade
        n = math.ceil(args.target / (L - d))
    else:
        # solve d exactly from N*L - N*d = target, with d in a sane range
        n, d = None, None
        max_cand = max(13, int(args.target / (L - 4.0)) + 2)
        for cand in range(2, max_cand):
            dd = (cand * L - args.target) / cand
            if 0.5 <= dd <= 4.0:  # long xfades are fine for water
                n, d = cand, dd
                break
        if n is None:  # degenerate: fixed 1.5s fades, accept ~target
            d = 1.5
            n = math.ceil(args.target / (L - d))
    total = n * L - n * d
    print(f"clip={L:.2f}s target={args.target}s -> {n} copies, xfade={d:.2f}s, out={total:.2f}s",
          flush=True)

    inputs = []
    for _ in range(n):
        inputs += ["-i", args.clip]
    parts, last, off = [], "[0:v]", 0.0
    for i in range(1, n):
        off = i * L - i * d
        tag = f"[x{i}]"
        parts.append(f"{last}[{i}:v]xfade=transition=fade:duration={d:.3f}:offset={off:.3f}{tag}")
        last = tag
    # loop seam: tail xfaded back into head
    # loop seam: tail xfaded back into head; xfade out = offset + head_len,
    # so offset must be len(tail) - d where tail = chain minus first d seconds
    chain_len = n * L - (n - 1) * d
    parts.append(f"{last}split[t][h]")
    parts.append(f"[h]trim=start=0:end={d:.3f},setpts=PTS-STARTPTS[head]")
    parts.append(f"[t]trim=start={d:.3f},setpts=PTS-STARTPTS[tail]")
    seam_off = chain_len - 2 * d
    parts.append(f"[tail][head]xfade=transition=fade:duration={d:.3f}:offset={seam_off:.3f},format=yuv420p[out]")
    fc = ";".join(parts)

    cmd = (["ffmpeg", "-y", "-v", "error"] + inputs +
           ["-filter_complex", fc, "-map", "[out]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-movflags", "+faststart", args.out])
    subprocess.run(cmd, check=True)
    print(f"wrote {args.out} ({probe_duration(args.out):.2f}s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
