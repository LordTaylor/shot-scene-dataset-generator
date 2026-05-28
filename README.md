# Shot/Scene Dataset Generator

A modular wildcard system for generating diverse character datasets with ComfyUI.  
Randomises shot types, poses, expressions, lighting, scenes and more — across **3.3 trillion** unique prompt combinations.

---

## Features

- **10 wildcard categories** — SHOT, POSE, EXPRESSION, LIGHTING, SCENE, CROWD, CREATURE, DYNAMIC, WEATHER, TIME
- **7 curated presets** — `full`, `action`, `environment`, `social`, `creature`, `chaos`, `minimal`
- **Pluggable character configs** — drop in any LoRA or base-model character
- **Reproducible seeds** — every image re-generable from its seed number
- **Dry-run mode** — preview prompts without submitting to ComfyUI
- **Clean public API** — import `build_wildcard_block()` for your own scripts

---

## Quick Start

### 1. Requirements

- Python 3.8+
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) running locally or on a server
- `requests` library

```bash
pip install requests
```

### 2. Clone

```bash
git clone https://github.com/YOUR_USERNAME/shot-scene-dataset-generator
cd shot-scene-dataset-generator
```

### 3. Create a character config

```bash
cp characters/my_character.py.example characters/hero.py
```

Edit `characters/hero.py` and fill in your character's description, checkpoint, and optional LoRA.

### 4. Point to your ComfyUI server

```bash
export COMFYUI_SERVER=http://192.168.1.100:8188
```

Or pass `--server http://localhost:8188` on the command line.

### 5. Generate

```bash
python3 gen_dataset.py --char hero --preset full --count 50
```

---

## Command Line Reference

```
python3 gen_dataset.py [options]

Required:
  --char NAME       Character config name (maps to characters/<name>.py)

Optional:
  --preset NAME     Wildcard preset: full | action | environment | social |
                    creature | chaos | minimal  (default: full)
  --count N         Number of images to generate (default: 20)
  --seed N          Starting seed — deterministic; omit for random
  --outdir PATH     Output folder prefix in ComfyUI output dir
  --server URL      ComfyUI server URL (default: $COMFYUI_SERVER or localhost:8188)
  --dry             Dry run — print prompts only, do not submit
```

### Examples

```bash
# Standard portrait dataset — 50 images, random seeds
python3 gen_dataset.py --char hero --preset full --count 50

# Action scenes only
python3 gen_dataset.py --char hero --preset action --count 30

# Preview prompts without generating anything
python3 gen_dataset.py --char hero --preset chaos --count 10 --dry

# Reproducible run from seed 1000
python3 gen_dataset.py --char hero --preset full --count 100 --seed 1000

# Remote server
python3 gen_dataset.py --char hero --count 20 --server http://192.168.1.50:8188
```

---

## Wildcard Categories

| Category   | Options | Description |
|------------|---------|-------------|
| SHOT       | 20      | Framing, angles, cinematic compositions |
| POSE       | 30      | Standing, action, sitting, relaxed |
| EXPRESSION | 17      | Emotions from confident to stoic |
| LIGHTING   | 20      | Natural, practical, magical, cinematic |
| SCENE      | 29      | Fantasy, urban, nature, dark environments |
| CROWD      | 15      | Social situations, background characters |
| CREATURE   | 18      | Fantasy mounts, monsters, companions |
| DYNAMIC    | 22      | Magic effects, combat FX, environmental |
| WEATHER    | 12      | Sky conditions, atmosphere |
| TIME       | 8       | Dawn through deep night |

### Presets

| Preset      | Categories included | Combinations |
|-------------|---------------------|--------------|
| `full`      | SHOT POSE EXPR LIGHT SCENE | 5,916,000 |
| `action`    | + DYNAMIC | 130,152,000 |
| `environment` | SHOT POSE SCENE WEATHER TIME | 1,670,400 |
| `social`    | SHOT POSE EXPR SCENE CROWD | 4,437,000 |
| `creature`  | SHOT POSE EXPR SCENE CREATURE LIGHT | 106,488,000 |
| `chaos`     | SHOT POSE EXPR LIGHT SCENE DYNAMIC WEATHER | 1,561,824,000 |
| `minimal`   | SHOT POSE SCENE | 17,400 |

---

## ComfyUI Node Layout

The generator builds this graph programmatically for each image:

```
┌──────────────────────────────────────────────────────────────────┐
│  CheckpointLoaderSimple                                          │
│   ckpt_name: your_model.safetensors                              │
└──────┬────────────────┬──────────────────────────────────────────┘
       │ MODEL          │ CLIP
       │                ▼
       │    ┌──────────────────────────────────────────────────┐
       │    │  CLIPTextEncode  [POSITIVE]                      │
       │    │   style + character description + wildcards      │
       │    └──────────────────┬───────────────────────────────┘
       │                       │ CONDITIONING
       │    ┌──────────────────────────────────────────────────┐
       │    │  CLIPTextEncode  [NEGATIVE]                      │
       │    │   negative prompt from character config          │
       │    └──────────────────┬───────────────────────────────┘
       │                       │ CONDITIONING
       │    ┌──────────────────────────────────────────────────┐
       │    │  EmptyLatentImage                                │
       │    │   width × height from character config           │
       │    └──────────────────┬───────────────────────────────┘
       │                       │ LATENT
       ▼                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  KSampler                                                        │
│   seed: auto-increments each image                               │
│   steps / cfg / sampler / scheduler: from character config       │
└──────┬───────────────────────────────────────────────────────────┘
       │ LATENT
       ▼
┌─────────────────┐      ┌───────────────────────────────────────┐
│  VAEDecode      │─────▶│  SaveImage                            │
│                 │      │   filename_prefix: dataset/char_preset │
└─────────────────┘      └───────────────────────────────────────┘
```

**With LoRA**, a `LoraLoader` is inserted between `CheckpointLoaderSimple` and `KSampler`, and the CLIP path is rewired through it automatically.

---

## Character Config Format

```python
# characters/my_character.py
CHARACTER = {
    # Trigger word for LoRA (or plain description if no LoRA)
    "trigger": "my_char_trigger",

    # Full character description injected into every prompt
    "description": (
        "my_char_trigger, long silver hair, blue eyes, "
        "red jacket, 1girl, solo, female,"
    ),

    # Style tags — applied before the character description
    "style": "best quality, masterpiece, cel-shaded,",

    # Negative prompt
    "negative": (
        "(worst quality, low quality, blurry), "
        "extra limbs, bad anatomy, watermark, text"
    ),

    # LoRA — set to None if not using one
    "lora":          "MyChar/my_char_v1.safetensors",
    "lora_strength": 0.75,

    # Checkpoint
    "checkpoint": "Toon/myModel_v1.safetensors",

    # Image dimensions and sampler settings
    "width":     768,
    "height":    1024,
    "steps":     28,
    "cfg":       7.0,
    "sampler":   "euler_ancestral",
    "scheduler": "karras",
}
```

---

## Using wildcards_db Standalone

```python
from wildcards_db import build_wildcard_block, WILDCARDS, PRESETS, pick, stats

# Pick one random item from a category
shot = pick(WILDCARDS["SHOT"])
# → "low angle shot, looking up at character"

# Build a full wildcard block for a preset
picked, prompt_fragment, label = build_wildcard_block(preset="action", seed=42)
# picked     → {"SHOT": "full body shot...", "POSE": "jumping...", ...}
# prompt_fragment → "full body shot, head to toe, jumping, airborne, ..."
# label      → "full_body_shot_jumping" (safe for filenames)

# Print database statistics
stats()
```

---

## Extending the Database

Add entries to any category in `wildcards_db.py`:

```python
WILDCARDS["SCENE"].append("secret laboratory, neon consoles, holographic displays")
WILDCARDS["DYNAMIC"].append("water splash, waves frozen in time")
```

Add a new category entirely:

```python
WILDCARDS["OUTFIT"] = [
    "battle armor, dented pauldrons, war-worn",
    "elegant ball gown, silk, intricate embroidery",
    "casual streetwear, hoodie, sneakers",
]

# Then add it to a preset
PRESETS["outfit_focus"] = ["SHOT", "POSE", "OUTFIT", "SCENE", "LIGHTING"]
```

---

## Workflow for LoRA Training

1. **Generate dataset** (60–150 images, mixed presets)
2. **Caption** with WD14 tagger or Florence-2 in ComfyUI
3. **Train** with Kohya or similar — keep trigger word consistent
4. **Update** your character config with the trained LoRA path
5. **Re-generate** at higher seed range to test

Full guide: [docs/lora_training.md](docs/lora_training.md)

---

## License

MIT — use freely, no attribution required.

---

## Contributing

PRs welcome for new wildcard entries, presets, or documentation improvements.  
Open an issue to discuss larger changes.
