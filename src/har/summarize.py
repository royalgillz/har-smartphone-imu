"""
Turn a recording's per-window predictions into a natural-language activity log
using the Claude API.

The CNN-LSTM still does the recognition; Claude only narrates the timeline and
flags low-confidence stretches. This is the optional "summary layer" -- it needs
an Anthropic API key in the ANTHROPIC_API_KEY environment variable.

Run:
    python -m src.har.summarize path/to/recording.zip
    python -m src.har.summarize            # first zip in data/test_zips
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

from . import config
from .predict import load_stats, load_model, extract_zip, window_predictions

MODEL = "claude-opus-4-8"

SYSTEM = (
    "You are a fitness/activity assistant. You are given a timeline of activity "
    "segments inferred from a smartphone's motion sensors by a CNN-LSTM classifier. "
    "Write a short, friendly activity log for the user: what they did, in order, "
    "with durations. Group trivially short blips into the surrounding activity. "
    "Explicitly flag any segment the model was unsure about (low confidence) and "
    "say the label there may be unreliable. Do not invent activities not in the "
    "timeline. Keep it to a few sentences plus a compact bulleted timeline."
)


def build_segments(preds, probs, labels):
    """Collapse consecutive same-label windows into timed segments."""
    step_s = config.STEP / config.SAMPLE_RATE_HZ
    win_s = config.WINDOW_SIZE / config.SAMPLE_RATE_HZ

    segments = []
    start = 0
    for i in range(1, len(preds) + 1):
        if i == len(preds) or preds[i] != preds[start]:
            idx = range(start, i)
            cls = int(preds[start])
            conf = float(np.mean([probs[j, cls] for j in idx]))
            t0 = start * step_s
            t1 = (i - 1) * step_s + win_s
            segments.append({
                "activity": str(labels[cls]),
                "start_s": round(t0, 1),
                "end_s": round(t1, 1),
                "duration_s": round(t1 - t0, 1),
                "confidence": round(conf, 2),
            })
            start = i
    return segments


def timeline_text(segments):
    lines = []
    for s in segments:
        lines.append(
            f"- {s['start_s']:.0f}s-{s['end_s']:.0f}s ({s['duration_s']:.0f}s): "
            f"{s['activity']} (confidence {s['confidence']:.2f})"
        )
    return "\n".join(lines)


def summarize(segments):
    try:
        import anthropic
    except ImportError:
        raise SystemExit("The 'anthropic' package is not installed. "
                         "Run: uv pip install --system-certs anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to use the summary layer.")

    prompt = (
        "Here is the inferred activity timeline for one recording. Confidence is "
        "0-1; treat anything below 0.6 as unreliable.\n\n"
        + timeline_text(segments)
    )
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        thinking={"type": "adaptive"},
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", help="zip or extracted folder; "
                    "default = first zip in data/test_zips")
    args = ap.parse_args()

    means, stds, labels = load_stats()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(len(labels), device)

    if args.target:
        target = Path(args.target)
        folder = extract_zip(target) if target.suffix.lower() == ".zip" else target
    else:
        zips = sorted(config.TEST_ZIP_DIR.glob("*.zip"))
        if not zips:
            raise SystemExit(f"No zips in {config.TEST_ZIP_DIR}")
        target = zips[0]
        folder = extract_zip(target)

    out = window_predictions(folder, model, means, stds, device)
    if out is None:
        raise SystemExit("No usable windows in this recording.")
    preds, probs = out
    segments = build_segments(preds, probs, labels)

    print(f"\nTimeline for {target.name}:")
    print(timeline_text(segments))
    print("\n--- Claude activity log ---")
    print(summarize(segments))


if __name__ == "__main__":
    main()
