#!/usr/bin/env python3
"""koryuai-easymusicviz: one CLI for every Koryuai visual engine.

Subcommands:
  liquid   audio-reactive "liquid metal" render from a song (engines/liquid.py)
  still    photorealistic still from a prompt via FLUX.1-schnell (engines/still.py)
  animate  still -> short clip via Wan 2.2 image-to-video (engines/animate.py)
  weave    weave a clip into a seamless loop of exact length (engines/weave.py)

Each engine is also runnable standalone; this CLI just puts them in one
place so a future GUI can drive the same entry points.
"""
import argparse, subprocess, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
ENG = os.path.join(HERE, "engines")

def run(script, args):
    cmd = [sys.executable, os.path.join(ENG, script)] + args
    r = subprocess.run(cmd)
    sys.exit(r.returncode)

def main():
    ap = argparse.ArgumentParser(prog="app.py",
        description="Koryuai EasyMusicViz — every visual engine, one CLI.")
    sub = ap.add_subparsers(dest="mode", required=True)

    for name, script, help_ in [
        ("liquid", "liquid.py", "audio-reactive liquid render from a song"),
        ("still", "still.py", "photorealistic still from a prompt"),
        ("animate", "animate.py", "animate a still into a short clip"),
        ("weave", "weave.py", "weave a clip into a seamless loop"),
    ]:
        p = sub.add_parser(name, help=help_,
            description=f"Forwards all extra args to engines/{script}")
        p.add_argument("rest", nargs=argparse.REMAINDER,
            help=f"arguments for engines/{script} (-- --help to see them)")

    a = ap.parse_args()
    rest = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
    scripts = {"liquid": "liquid.py", "still": "still.py",
               "animate": "animate.py", "weave": "weave.py"}
    run(scripts[a.mode], rest)

if __name__ == "__main__":
    main()
