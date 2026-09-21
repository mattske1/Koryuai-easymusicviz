#!/usr/bin/env python3
"""Stillwater stage 1 — generate a photorealistic long-exposure still.

Uses the free FLUX.1-schnell HuggingFace Space (black-forest-labs/FLUX.1-schnell).
No token required for the public Space; set HF_TOKEN if you have one and want
a shorter queue.

    python3 still.py --prompt "..." --out still.png
"""
import argparse
import os
import shutil
import sys

# gradio_client chokes on bracketed-IPv6 no_proxy entries on some hosts
# (Invalid port: ':1]'), so they are popped before connecting
os.environ.pop("no_proxy", None)
os.environ.pop("NO_PROXY", None)

from gradio_client import Client  # noqa: E402

SPACE = "black-forest-labs/FLUX.1-schnell"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, help="scene description")
    ap.add_argument("--out", default="still.png")
    ap.add_argument("--width", type=int, default=1216)
    ap.add_argument("--height", type=int, default=704)
    ap.add_argument("--seed", type=int, default=7,
                    help="0 = randomize each run")
    args = ap.parse_args()

    client = Client(SPACE, token=os.environ.get("HF_TOKEN"),
                    httpx_kwargs={"timeout": 600})
    print(f"generating still via {SPACE} ...", flush=True)
    result = client.predict(
        api_name="/infer",
        prompt=args.prompt,
        seed=args.seed,
        randomize_seed=(args.seed == 0),
        width=args.width,
        height=args.height,
        num_inference_steps=4,
    )
    img = result[0] if isinstance(result, (list, tuple)) else result
    if isinstance(img, dict):
        img = img.get("url") or img.get("path", img)
    if isinstance(img, str) and img.startswith("http"):
        import urllib.request
        req = urllib.request.Request(img)
        with urllib.request.urlopen(req, timeout=300) as r, \
                open(args.out, "wb") as f:
            shutil.copyfileobj(r, f)
    else:
        shutil.copy(img, args.out)
    print(f"wrote {args.out} ({args.width}x{args.height}, seed {args.seed})",
          flush=True)


if __name__ == "__main__":
    main()
