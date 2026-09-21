#!/usr/bin/env python3
"""Stillwater stage 2 — animate a still into a short ambient clip.

Uses the free Wan 2.2 image-to-video HuggingFace Space
(Upsampler/wan-2-2-14b-image-to-video). No token required, but the free
queue can take a few minutes.

Gotchas baked in:
  - the VM's bracketed-IPv6 no_proxy entries crash gradio_client
    (Invalid port: ':1]'), so they are popped before connecting;
  - image inputs must be FileData dicts, or the server 500s.

    python3 animate.py --still still.png --out clip.mp4
"""
import argparse
import os
import shutil
import sys

# gradio_client chokes on bracketed-IPv6 no_proxy entries on some hosts
os.environ.pop("no_proxy", None)
os.environ.pop("NO_PROXY", None)

from gradio_client import Client  # noqa: E402

SPACE = "Upsampler/wan-2-2-14b-image-to-video"

DEFAULT_ANIMATE_PROMPT = (
    "Animate this still image with continuous natural motion: water flowing, "
    "mist drifting, light shimmering on the surface. Static camera, tranquil "
    "and meditative, seamless ambient motion with no cuts."
)

DEFAULT_NEGATIVE = (
    "oversaturated, overexposed, static, blurry, text, watermark, "
    "worst quality, low quality"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", required=True, help="input still image")
    ap.add_argument("--prompt", default=DEFAULT_ANIMATE_PROMPT)
    ap.add_argument("--out", default="clip.mp4")
    ap.add_argument("--duration", type=float, default=5.0,
                    help="clip seconds (0.5-5.0)")
    ap.add_argument("--steps", type=int, default=6, help="denoise steps (1-12)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if not 0.5 <= args.duration <= 5.0:
        sys.exit("duration must be between 0.5 and 5.0 seconds")

    client = Client(SPACE, token=os.environ.get("HF_TOKEN"))
    print("animating (a few minutes on the free queue) ...", flush=True)
    img = {"path": os.path.abspath(args.still),
           "meta": {"_type": "gradio.FileData"}}
    video_path, used_seed = client.predict(
        input_image=img,
        prompt=args.prompt,
        steps=args.steps,
        negative_prompt=DEFAULT_NEGATIVE,
        duration_seconds=args.duration,
        guidance_scale=1.0,
        guidance_scale_2=1.0,
        seed=args.seed,
        randomize_seed=False,
        end_image=None,
        api_name="/generate_video",
    )
    shutil.copy(video_path, args.out)
    print(f"wrote {args.out} (seed {int(used_seed)})", flush=True)


if __name__ == "__main__":
    sys.exit(main())
