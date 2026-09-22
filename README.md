# Koryuai EasyMusicViz

Every Koryuai visual engine in one place — a composite of everything we've
open-sourced, and the home for every texture we build next.

Koryuai EasyMusicViz is a product of **Koryuai**, written by **Mattske** with
agentic support for **Edged Out Records**, and given away for free here for
independent artists looking to create quick visuals for their music to make it
eye-catching on YouTube and social media platforms.

## The engines

| Mode | Engine | What it does |
|---|---|---|
| `liquid` | audio-reactive "liquid metal" | Analyzes your song's energy, brightness, and pulse, then renders a flowing turbulent field that breathes with the music. 100% programmatic — runs anywhere, no AI services. |
| `galaxy` | audio-reactive deep space | A spiral galaxy with domain-warped nebula clouds, a twinkling parallax starfield, and a pulse-flaring core — all driven by your song. 100% programmatic, no AI services. |
| `still` | FLUX.1-schnell | Photorealistic stills (water, fire, whatever you prompt) from a free hosted Space. |
| `animate` | Wan 2.2 image-to-video | Brings a still to life with a static camera, from a free hosted Space. |
| `weave` | crossfade weaver | Chains any clip into a seamless loop of exact length. The weave doesn't care where the clip came from. |

Prompts for the still/animate/weave pipeline live in `scenes/` — water and
fire scenes to start, more textures as we build them.

## Quickstart

```bash
pip install -r requirements.txt
# ffmpeg must be installed: https://ffmpeg.org/download.html
# The still/animate steps use free HuggingFace Spaces — no accounts or keys.
# (Set HF_TOKEN if you have one and want a shorter animation queue.)

# Audio-reactive liquid render of your song (full length, audio muxed in)
python3 app.py liquid -- song.wav --out visuals.mp4 --full-length

# 10-second seamless liquid loop for Shorts/Reels/TikTok
python3 app.py liquid -- song.wav --out loop.mp4 --seconds 10

# Audio-reactive galaxy render (full length, audio muxed in)
python3 app.py galaxy -- song.wav --out galaxy.mp4 --full-length --palette nebula

# Ambient water/fire loop: still -> animate -> weave
python3 app.py still -- --prompt "$(head -1 scenes/fire-pit-night.txt)" --out still.png
python3 app.py animate -- --still still.png --out clip.mp4
python3 app.py weave -- --clip clip.mp4 --target 60 --out loop-60s.mp4
```

Each engine also runs standalone from `engines/` — `app.py` just forwards
your args, so everything you learn transfers.

**A note on quality:** the still and animation engines are wired to free public
inference endpoints, so they're rate-limited and the output is honestly
demo-grade — sometimes somewhat garbage. That's the free tier, not the
pipeline. If you wire `still.py` / `animate.py` to your own model or paid
credits, quality improves tremendously. Each engine is one file with one job
(still in → clip out), so swapping in your own is straightforward. The weave
step doesn't care where the clip came from.

## Roadmap

The plan is a GUI on top of this same CLI: load up your song, pick a material
(liquid, galaxy, water, fire, future textures), tune properties (palette, turbulence,
duration, camera behavior), tick output formats (16:9 for YouTube, 9:16 for
Reels/TikTok/Shorts, 1:1 for the feed), and render — same engines underneath,
no new pipeline to learn.

## License

MIT — free for whatever you want to make with it.
