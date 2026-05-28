#!/usr/bin/env python3
"""
=============================================================
  Shot/Scene Dataset Generator — ComfyUI batch image gen
  Uses wildcards_db.py for shot/scene/pose/lighting variety
=============================================================
  Usage:
    python3 gen_dataset.py --char my_character --preset full --count 50
    python3 gen_dataset.py --char my_character --preset action --count 30
    python3 gen_dataset.py --char my_character --preset minimal --count 20 --dry

  Set COMFYUI_SERVER env var or pass --server:
    export COMFYUI_SERVER=http://192.168.1.100:8188
    python3 gen_dataset.py --char my_character --count 50
=============================================================
"""
import os
import sys
import requests
import uuid
import time
import json
import argparse
import random
import importlib.util

from wildcards_db import build_wildcard_block, WILDCARDS, PRESETS

# ---------------------------------------------------------------------------
_SERVER = os.environ.get("COMFYUI_SERVER", "http://localhost:8188")


def get_server():
    return _SERVER


# ---------------------------------------------------------------------------
def load_character(name: str, chars_dir: str = None) -> dict:
    """
    Load character config from <chars_dir>/<name>.py
    Defaults to characters/ subfolder next to this script.
    The file must define a dict called CHARACTER.
    """
    if chars_dir is None:
        chars_dir = os.path.join(os.path.dirname(__file__), "characters")
    path = os.path.join(chars_dir, f"{name}.py")
    if not os.path.exists(path):
        example = os.path.join(os.path.dirname(__file__), "characters", "my_character.py.example")
        sys.exit(
            f"[ERROR] Character '{name}' not found at: {path}\n"
            f"Copy the example and fill it in:\n  cp {example} {chars_dir}/{name}.py"
        )
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "CHARACTER"):
        sys.exit(f"[ERROR] {path} must define a 'CHARACTER' dict.")
    return mod.CHARACTER


# ---------------------------------------------------------------------------
def build_api(char: dict, preset: str, seed: int, output_prefix: str):
    """Build a ComfyUI API-format payload for a single generation job."""
    picked, wildcard_fragment, label = build_wildcard_block(preset=preset, seed=seed)

    # Full positive prompt: style + character description + wildcards
    positive = f"{char['style']} {char['description']} {wildcard_fragment}"
    negative = char["negative"]

    api = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": char["checkpoint"]}
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive, "clip": ["1", 1]}
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative, "clip": ["1", 1]}
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width":      char["width"],
                "height":     char["height"],
                "batch_size": 1
            }
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model":        ["1", 0],
                "positive":     ["2", 0],
                "negative":     ["3", 0],
                "latent_image": ["4", 0],
                "seed":                   seed,
                "control_after_generate": "fixed",
                "steps":      char["steps"],
                "cfg":        char["cfg"],
                "sampler_name": char["sampler"],
                "scheduler":    char["scheduler"],
                "denoise":      1.0,
            }
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]}
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {
                "images":          ["6", 0],
                "filename_prefix": output_prefix
            }
        },
    }

    # Inject LoRA when defined in character config
    if char.get("lora"):
        api["10"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model":          ["1", 0],
                "clip":           ["1", 1],
                "lora_name":      char["lora"],
                "strength_model": char.get("lora_strength", 1.0),
                "strength_clip":  char.get("lora_strength", 1.0),
            }
        }
        api["2"]["inputs"]["clip"]    = ["10", 1]
        api["3"]["inputs"]["clip"]    = ["10", 1]
        api["5"]["inputs"]["model"]   = ["10", 0]

    return api, picked, label


# ---------------------------------------------------------------------------
def submit(api: dict) -> str:
    cid = str(uuid.uuid4())
    r = requests.post(
        f"{get_server()}/prompt",
        json={"prompt": api, "client_id": cid},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["prompt_id"]


def poll(pid: str, timeout: int = 600):
    deadline = time.time() + timeout
    t0 = time.time()
    while time.time() < deadline:
        try:
            h = requests.get(f"{get_server()}/history/{pid}", timeout=60).json()
            if not isinstance(h, dict):
                time.sleep(8)
                continue
        except Exception:
            time.sleep(10)
            continue
        if pid in h:
            entry = h[pid]
            if not isinstance(entry, dict):
                time.sleep(8)
                continue
            st = entry.get("status", {})
            if isinstance(st, dict) and st.get("status_str") == "error":
                raise RuntimeError(str(st.get("messages", [])))
            imgs = [
                img["filename"]
                for _, out in entry.get("outputs", {}).items()
                for img in (out.get("images", []) if isinstance(out, dict) else [])
            ]
            if imgs:
                return imgs, time.time() - t0
        time.sleep(8)
    raise TimeoutError(f"Timed out after {timeout}s")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Shot/Scene Dataset Generator — ComfyUI batch generation"
    )
    parser.add_argument(
        "--char", required=True,
        help="Character config name (maps to characters/<name>.py)"
    )
    parser.add_argument(
        "--preset", default="full",
        choices=list(PRESETS.keys()),
        help="Wildcard preset (default: full)"
    )
    parser.add_argument(
        "--count", type=int, default=20,
        help="Number of images to generate (default: 20)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Starting seed (random if not set)"
    )
    parser.add_argument(
        "--outdir", default=None,
        help="Output folder prefix inside ComfyUI output (default: dataset/<char>_<preset>)"
    )
    parser.add_argument(
        "--server", default=None,
        help="ComfyUI server URL (default: $COMFYUI_SERVER or localhost:8188)"
    )
    parser.add_argument(
        "--chars-dir", default=None,
        help="Path to folder with character .py configs (default: characters/ next to this script)"
    )
    parser.add_argument(
        "--dry", action="store_true",
        help="Dry run — show prompts without submitting"
    )
    args = parser.parse_args()

    if args.server:
        globals()["_SERVER"] = args.server

    char   = load_character(args.char, args.chars_dir)
    outdir = args.outdir or f"dataset/{args.char}_{args.preset}"
    start_seed = args.seed if args.seed is not None else random.randint(1, 999_999)

    cats         = PRESETS[args.preset]
    total_combos = 1
    for c in cats:
        total_combos *= len(WILDCARDS[c])

    print(f"=== DATASET GENERATOR ===")
    print(f"  Character : {args.char}")
    print(f"  Preset    : {args.preset}  ({len(cats)} categories, {total_combos:,} combos)")
    print(f"  Count     : {args.count}")
    print(f"  Seed start: {start_seed}")
    print(f"  Server    : {get_server()}")
    print(f"  Output    : ComfyUI/output/{outdir}/")
    print()

    ok, failed = 0, []
    elapsed_times = []

    for i in range(args.count):
        seed = start_seed + i
        api, picked, label = build_api(char, args.preset, seed, outdir)

        if args.dry:
            print(f"[{i+1:03d}] seed={seed}  label={label}")
            for cat, val in picked.items():
                print(f"       {cat.lower()}: {val}")
            print()
            continue

        print(f"[{i+1:03d}/{args.count}] {label[:45]:<45}  seed={seed}", end="  ", flush=True)
        try:
            pid = submit(api)
            print(f"→ {pid[:8]}...", end=" ", flush=True)
            imgs, elapsed = poll(pid)
            mark = "🔥" if elapsed < 60 else "⏳"
            print(f"{mark} {elapsed:.0f}s → {imgs[0] if imgs else '?'}")
            ok += 1
            if i > 0:
                elapsed_times.append(elapsed)
        except Exception as e:
            print(f"❌ {e}")
            failed.append((seed, label, str(e)))

        # ETA every 10 images
        if (i + 1) % 10 == 0:
            remaining = args.count - (i + 1)
            avg = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 30
            print(f"\n  ── {ok}/{i+1} ✅  ETA: ~{remaining * avg / 60:.1f} min ──\n")

    if not args.dry:
        print(f"\n=== DONE: {ok}/{args.count} ===")
        if elapsed_times:
            print(f"  Avg time : {sum(elapsed_times)/len(elapsed_times):.0f}s / image")
        if failed:
            print(f"\n  Failed ({len(failed)}):")
            for s, l, e in failed:
                print(f"    seed={s}  {l}: {e[:80]}")
        print(f"\n  Output: ComfyUI/output/{outdir}/")


if __name__ == "__main__":
    main()
