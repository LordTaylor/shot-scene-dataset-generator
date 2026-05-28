import os
import random
import importlib.util

try:
    from .wildcards_db import build_wildcard_block, WILDCARDS, PRESETS
except ImportError:
    from wildcards_db import build_wildcard_block, WILDCARDS, PRESETS

try:
    import folder_paths as _folder_paths
    import comfy.sd as _comfy_sd
    import comfy.utils as _comfy_utils
    _COMFYUI_AVAILABLE = True
except ImportError:
    _folder_paths = None
    _comfy_sd = None
    _comfy_utils = None
    _COMFYUI_AVAILABLE = False


# ── Pure helpers ─────────────────────────────────────────────────────────────

def _apply_loras(model, clip, pairs: list, loader=None, path_resolver=None):
    """Apply (lora_name, strength) pairs sequentially. Skips entries named 'None'."""
    if loader is None:
        def loader(m, c, lora_path, sm, sc):
            lora = _comfy_utils.load_torch_file(lora_path, safe_load=True)
            return _comfy_sd.load_lora_for_models(m, c, lora, sm, sc)
    if path_resolver is None:
        path_resolver = lambda name: _folder_paths.get_full_path("loras", name)

    for name, strength in pairs:
        if name == "None":
            continue
        model, clip = loader(model, clip, path_resolver(name), strength, strength)
    return model, clip

def _build_prompt(text: str, preset: str, seed: int) -> str:
    _, fragment, _ = build_wildcard_block(preset=preset, seed=seed)
    clean = text.strip().rstrip(", ")
    if clean:
        return f"{clean}, {fragment}"
    return fragment


def _build_prompt_cats(text: str, categories: dict, seed: int,
                       overrides: dict = None) -> str:
    """Build prompt from category toggles.

    overrides = {CAT: fixed_string} bypasses random pick for that category.
    """
    rng = random.Random(seed)
    picks = []
    for cat in WILDCARDS:
        if not categories.get(cat, False):
            continue
        if overrides and cat in overrides:
            picks.append(overrides[cat])
        else:
            picks.append(rng.choice(WILDCARDS[cat]))
    fragment = ", ".join(picks)
    clean = text.strip().rstrip(", ")
    if clean and fragment:
        return f"{clean}, {fragment}"
    return clean or fragment


def _append_to_db(category: str, new_entries: list, db_path: str = None) -> int:
    if db_path is None:
        db_path = os.path.join(os.path.dirname(__file__), "wildcards_db.py")

    spec = importlib.util.spec_from_file_location("_wdb_check", db_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    current = set(mod.WILDCARDS.get(category, []))
    to_add = [e for e in new_entries if e not in current]

    if not to_add:
        return 0

    with open(db_path, "r") as f:
        lines = f.readlines()

    in_section = False
    insert_idx = None
    for i, line in enumerate(lines):
        if f'WILDCARDS["{category}"]' in line and "= [" in line:
            in_section = True
            continue
        if in_section and line.strip() == "]":
            insert_idx = i
            break

    if insert_idx is None:
        raise ValueError(f"Category '{category}' not found in {db_path}")

    new_lines = [f'    "{e}",\n' for e in to_add]
    lines[insert_idx:insert_idx] = new_lines

    with open(db_path, "w") as f:
        f.writelines(lines)

    return len(to_add)


# ── ComfyUI nodes ─────────────────────────────────────────────────────────────

class UltraWildcardNode:
    # Default on: basic framing categories
    _DEFAULT_ON = {"SHOT", "POSE", "SCENE"}

    _SHOT_LOCKS = {
        "full body":  "full body shot, head to toe",
        "half body":  "waist up shot, half body",
        "portrait":   "close up portrait, face focus",
    }

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "clip":        ("CLIP",),
            "text":        ("STRING", {"multiline": True, "default": ""}),
            "seed":        ("INT", {"default": -1, "min": -1, "max": 0x7FFFFFFFFFFFFFFF}),
            "enabled_all": ("BOOLEAN", {"default": True}),
        }
        for cat in WILDCARDS:
            required[cat.lower()] = ("BOOLEAN", {"default": cat in cls._DEFAULT_ON})
            if cat == "SHOT":
                required["shot_lock"] = (["random"] + list(cls._SHOT_LOCKS.keys()),)
        return {"required": required}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION     = "encode"
    CATEGORY     = "conditioning/wildcards"

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        # seed=-1 means "random every run" — return NaN so ComfyUI never caches
        if seed == -1:
            return float("nan")
        return seed

    def encode(self, clip, text, seed, enabled_all, shot_lock="random", **kwargs):
        if seed == -1:
            seed = random.randint(0, 0x7FFFFFFFFFFFFFFF)
        # enabled_all=True  → force all categories on, ignore individual toggles
        # enabled_all=False → use individual category toggles
        if enabled_all:
            categories = {cat: True for cat in WILDCARDS}
        else:
            categories = {cat: kwargs.get(cat.lower(), False) for cat in WILDCARDS}
        overrides = {}
        if shot_lock != "random" and shot_lock in self._SHOT_LOCKS:
            overrides["SHOT"] = self._SHOT_LOCKS[shot_lock]
        full_prompt = _build_prompt_cats(text, categories, seed, overrides=overrides)
        tokens = clip.tokenize(full_prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]],)


class WildcardAppend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "category": (list(WILDCARDS.keys()),),
                "entries":  ("STRING", {
                    "multiline": True,
                    "default": "new entry 1 | new entry 2 | new entry 3",
                }),
                "save":     ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE  = True
    FUNCTION     = "append"
    CATEGORY     = "conditioning/wildcards"

    def append(self, category, entries, save):
        if not save:
            return ({},)
        new_entries = [e.strip() for e in entries.split("|") if e.strip()]
        if not new_entries:
            print("[WildcardAppend] No entries provided.")
            return ({},)
        added = _append_to_db(category, new_entries)
        print(f"[WildcardAppend] Added {added} new entries to {category}.")
        return ({},)


class MultiLoraLoader:
    _MAX_SLOTS = 8

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"]
        if _COMFYUI_AVAILABLE:
            loras += _folder_paths.get_filename_list("loras")
        s = {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}
        optional = {}
        for i in range(1, cls._MAX_SLOTS + 1):
            optional[f"lora_{i}"]     = (loras,)
            optional[f"strength_{i}"] = ("FLOAT", s)
            optional[f"enabled_{i}"]  = ("BOOLEAN", {"default": True})
        return {
            "required": {
                "model":       ("MODEL",),
                "clip":        ("CLIP",),
                "enabled_all": ("BOOLEAN", {"default": True}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION     = "load_loras"
    CATEGORY     = "conditioning/wildcards"

    def load_loras(self, model, clip, enabled_all, **kwargs):
        if not enabled_all:
            return (model, clip)
        pairs = [
            (kwargs[f"lora_{i}"], kwargs[f"strength_{i}"])
            for i in range(1, self._MAX_SLOTS + 1)
            if kwargs.get(f"enabled_{i}", True)
            and kwargs.get(f"lora_{i}", "None") != "None"
        ]
        model, clip = _apply_loras(model, clip, pairs)
        return (model, clip)
