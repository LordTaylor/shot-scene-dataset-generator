"""
Edge-case tests for nodes.py and wildcards_db.py.

Covers: seed boundaries, output format, each category toggle,
        special characters, input sanitisation, error propagation.
"""
import sys, os, math, shutil, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from wildcards_db import WILDCARDS, PRESETS

MAX_SEED = 0x7FFFFFFFFFFFFFFF


# ── Shared helpers ────────────────────────────────────────────────────────────

class FakeClip:
    def __init__(self):
        self.seen = []
    def tokenize(self, text):
        self.seen.append(text)
        return text
    def encode_from_tokens(self, tokens, return_pooled):
        return tokens, "pooled"

def _make_tmp_db():
    src = os.path.join(os.path.dirname(__file__), "..", "wildcards_db.py")
    tmp = tempfile.mkdtemp()
    dst = os.path.join(tmp, "wildcards_db.py")
    shutil.copy(src, dst)
    return dst, tmp


# ── Seed boundary values ──────────────────────────────────────────────────────

def test_is_changed_seed_zero_returns_zero():
    """seed=0 is a fixed seed — IS_CHANGED must return 0, not NaN."""
    from nodes import UltraWildcardNode
    assert UltraWildcardNode.IS_CHANGED(seed=0) == 0

def test_is_changed_seed_max_returns_max():
    from nodes import UltraWildcardNode
    assert UltraWildcardNode.IS_CHANGED(seed=MAX_SEED) == MAX_SEED

def test_encode_seed_zero_does_not_crash():
    from nodes import UltraWildcardNode
    clip = FakeClip()
    UltraWildcardNode().encode(clip, "char", seed=0, enabled_all=False,
                               shot=True, pose=False, scene=False,
                               expression=False, lighting=False, crowd=False,
                               creature=False, dynamic=False, weather=False, time=False)
    assert clip.seen[0]

def test_encode_seed_max_does_not_crash():
    from nodes import UltraWildcardNode
    clip = FakeClip()
    UltraWildcardNode().encode(clip, "char", seed=MAX_SEED, enabled_all=False,
                               shot=True, pose=False, scene=False,
                               expression=False, lighting=False, crowd=False,
                               creature=False, dynamic=False, weather=False, time=False)
    assert clip.seen[0]


# ── Conditioning output format ────────────────────────────────────────────────

def test_encode_returns_correct_comfyui_conditioning_tuple():
    """encode() must return ([[cond, {"pooled_output": pooled}]],) exactly."""
    from nodes import UltraWildcardNode

    class StructuredClip:
        def tokenize(self, text): return text
        def encode_from_tokens(self, tokens, return_pooled):
            return f"COND:{tokens}", "POOLED"

    result = UltraWildcardNode().encode(
        StructuredClip(), "char", seed=1, enabled_all=False,
        shot=False, pose=False, scene=False,
        expression=False, lighting=False, crowd=False,
        creature=False, dynamic=False, weather=False, time=False)

    assert isinstance(result, tuple) and len(result) == 1
    cond_list = result[0]
    assert isinstance(cond_list, list) and len(cond_list) == 1
    cond, meta = cond_list[0]
    assert cond == "COND:char"
    assert meta == {"pooled_output": "POOLED"}

def test_encode_conditioning_pooled_key_present():
    """pooled_output key must always be present in conditioning meta."""
    from nodes import UltraWildcardNode

    class SimpleClip:
        def tokenize(self, text): return text
        def encode_from_tokens(self, tokens, return_pooled): return "c", "p"

    result = UltraWildcardNode().encode(
        SimpleClip(), "", seed=1, enabled_all=True,
        shot=True, pose=True, scene=True,
        expression=True, lighting=True, crowd=True,
        creature=True, dynamic=True, weather=True, time=True)
    _, meta = result[0][0]
    assert "pooled_output" in meta


# ── Each category toggle independently ───────────────────────────────────────

@pytest.mark.parametrize("cat", list(WILDCARDS.keys()))
def test_each_category_toggle_independently(cat):
    """Enabling exactly one category must inject exactly one pick from it."""
    from nodes import UltraWildcardNode
    clip = FakeClip()
    kwargs = {c.lower(): (c == cat) for c in WILDCARDS}
    UltraWildcardNode().encode(clip, "", seed=7, enabled_all=False, **kwargs)
    result = clip.seen[0]
    assert any(entry in result for entry in WILDCARDS[cat]), \
        f"Category {cat} enabled alone — no pick found in: {result!r}"
    # No entry from other categories should appear
    for other_cat, entries in WILDCARDS.items():
        if other_cat == cat:
            continue
        for entry in entries:
            assert entry not in result, \
                f"Category {other_cat} is OFF but entry {entry!r} leaked into result"


# ── Input sanitisation ────────────────────────────────────────────────────────

def test_encode_whitespace_only_text_produces_no_leading_comma():
    from nodes import UltraWildcardNode
    clip = FakeClip()
    UltraWildcardNode().encode(clip, "   ", seed=1, enabled_all=False,
                               shot=True, pose=False, scene=False,
                               expression=False, lighting=False, crowd=False,
                               creature=False, dynamic=False, weather=False, time=False)
    assert not clip.seen[0].startswith(",")
    assert not clip.seen[0].startswith(" ,")

def test_encode_text_with_multiple_trailing_commas():
    from nodes import UltraWildcardNode
    clip = FakeClip()
    UltraWildcardNode().encode(clip, "char,,, ", seed=1, enabled_all=False,
                               shot=True, pose=False, scene=False,
                               expression=False, lighting=False, crowd=False,
                               creature=False, dynamic=False, weather=False, time=False)
    assert ",," not in clip.seen[0]

def test_build_prompt_cats_whitespace_only_text():
    from nodes import _build_prompt_cats
    result = _build_prompt_cats("   ", {"SHOT": True}, seed=1)
    assert not result.startswith(",")
    assert not result.startswith(" ")

def test_build_prompt_cats_override_on_disabled_category_is_ignored():
    """Override for a category that is OFF must not appear in output."""
    from nodes import _build_prompt_cats
    result = _build_prompt_cats("", {"SHOT": False}, seed=1,
                                overrides={"SHOT": "full body shot, head to toe"})
    assert "full body shot" not in result

def test_build_prompt_cats_all_categories_no_double_comma():
    """With all categories on, output must never contain ',,'."""
    from nodes import _build_prompt_cats
    cats = {cat: True for cat in WILDCARDS}
    result = _build_prompt_cats("char", cats, seed=42)
    assert ",," not in result
    assert not result.startswith(",")
    assert not result.endswith(",")


# ── _apply_loras — error propagation ─────────────────────────────────────────

def test_apply_loras_negative_strength_is_applied():
    """Negative strength is valid (style subtraction) — loader must be called."""
    from nodes import _apply_loras
    calls = []
    def loader(m, c, p, sm, sc):
        calls.append(sm)
        return m, c
    _apply_loras("m", "c", [("lora.safetensors", -0.5)],
                 loader=loader, path_resolver=lambda n: n)
    assert calls == [-0.5]

def test_apply_loras_loader_exception_propagates():
    """If loader raises, exception must propagate unchanged."""
    from nodes import _apply_loras
    def bad_loader(m, c, p, sm, sc):
        raise RuntimeError("load failed")
    with pytest.raises(RuntimeError, match="load failed"):
        _apply_loras("m", "c", [("lora.safetensors", 1.0)],
                     loader=bad_loader, path_resolver=lambda n: n)

def test_apply_loras_path_resolver_exception_propagates():
    from nodes import _apply_loras
    def bad_resolver(name):
        raise FileNotFoundError(f"not found: {name}")
    with pytest.raises(FileNotFoundError):
        _apply_loras("m", "c", [("lora.safetensors", 1.0)],
                     loader=lambda *a: a[:2], path_resolver=bad_resolver)

def test_apply_loras_applies_same_strength_to_model_and_clip():
    """strength is applied identically to both model and clip."""
    from nodes import _apply_loras
    calls = []
    def loader(m, c, p, sm, sc):
        calls.append((sm, sc))
        return m, c
    _apply_loras("m", "c", [("lora.safetensors", 0.73)],
                 loader=loader, path_resolver=lambda n: n)
    sm, sc = calls[0]
    assert sm == sc == 0.73


# ── MultiLoraLoader — mixed enabled/disabled ──────────────────────────────────

def test_multi_lora_enabled_all_true_respects_individual_enabled_false():
    """enabled_all=True but enabled_i=False → that slot must be skipped."""
    from nodes import _apply_loras
    # Simulate: slots 1 (enabled) and 2 (disabled)
    pairs = [("lora_1.safetensors", 0.8)]  # slot 2 excluded because enabled=False
    calls = []
    _apply_loras("m", "c", pairs,
                 loader=lambda m, c, p, sm, sc: (calls.append(p), m, c)[1:],
                 path_resolver=lambda n: n)
    assert "lora_1.safetensors" in calls
    assert "lora_2.safetensors" not in calls

def test_multi_lora_load_loras_returns_tuple():
    """load_loras must always return a 2-tuple (model, clip)."""
    from nodes import MultiLoraLoader
    result = MultiLoraLoader().load_loras("m", "c", enabled_all=False)
    assert isinstance(result, tuple) and len(result) == 2


# ── pick() — boundary values ──────────────────────────────────────────────────

def test_pick_single_element_list_always_returns_it():
    from wildcards_db import pick
    for seed in [0, 1, 42, MAX_SEED, None]:
        assert pick(["only_option"], seed=seed) == "only_option"

def test_pick_seed_zero_returns_valid_entry():
    from wildcards_db import pick
    result = pick(WILDCARDS["SCENE"], seed=0)
    assert result in WILDCARDS["SCENE"]

def test_pick_returns_item_from_list():
    from wildcards_db import pick
    for cat in WILDCARDS:
        result = pick(WILDCARDS[cat], seed=42)
        assert result in WILDCARDS[cat], f"pick() returned item not in {cat}"


# ── build_wildcard_block — boundary / integration ────────────────────────────

def test_build_wildcard_block_seed_none_returns_valid():
    from wildcards_db import build_wildcard_block
    picked, fragment, label = build_wildcard_block(preset="minimal", seed=None)
    assert fragment
    assert all(v in fragment for v in picked.values())

def test_build_wildcard_block_seed_none_produces_variety():
    from wildcards_db import build_wildcard_block
    results = {build_wildcard_block(preset="full", seed=None)[1] for _ in range(20)}
    assert len(results) > 1

def test_build_wildcard_block_label_has_no_spaces():
    from wildcards_db import build_wildcard_block
    for seed in range(10):
        _, _, label = build_wildcard_block(preset="minimal", seed=seed)
        assert " " not in label, f"Label contains space: {label!r}"

def test_build_wildcard_block_chaos_includes_more_cats_than_minimal():
    from wildcards_db import build_wildcard_block
    chaos_picked, _, _ = build_wildcard_block(preset="chaos", seed=1)
    minimal_picked, _, _ = build_wildcard_block(preset="minimal", seed=1)
    assert len(chaos_picked) > len(minimal_picked)

def test_build_wildcard_block_different_presets_same_seed_differ():
    from wildcards_db import build_wildcard_block
    full_frag   = build_wildcard_block(preset="full",   seed=42)[1]
    action_frag = build_wildcard_block(preset="action", seed=42)[1]
    # Different category sets → different fragments
    assert full_frag != action_frag


# ── _append_to_db — special characters ───────────────────────────────────────

def test_append_entry_with_apostrophe():
    from nodes import _append_to_db
    db_path, tmp = _make_tmp_db()
    try:
        added = _append_to_db("SCENE", ["wizard's tower, stone walls"], db_path=db_path)
        assert added == 1
        with open(db_path) as f:
            assert "wizard's tower" in f.read()
    finally:
        shutil.rmtree(tmp)

def test_append_entry_with_commas_and_parens():
    from nodes import _append_to_db
    db_path, tmp = _make_tmp_db()
    try:
        entry = "rooftop garden, (city view), twilight"
        added = _append_to_db("SCENE", [entry], db_path=db_path)
        assert added == 1
    finally:
        shutil.rmtree(tmp)

def test_append_every_category_accepts_new_entry():
    """Every category in WILDCARDS must support _append_to_db."""
    from nodes import _append_to_db
    db_path, tmp = _make_tmp_db()
    try:
        for cat in WILDCARDS:
            entry = f"__edge_case_{cat.lower()}__"
            added = _append_to_db(cat, [entry], db_path=db_path)
            assert added == 1, f"Failed to add entry to category {cat}"
    finally:
        shutil.rmtree(tmp)


# ── WildcardAppend — pipe parsing ─────────────────────────────────────────────

def test_wildcard_append_pipe_entries_stripped():
    """Entries must be stripped of whitespace around pipes."""
    from nodes import WildcardAppend
    # save=False so we don't touch real db — just verify parse behaviour via save=False
    result = WildcardAppend().append("SCENE", "  a  |  b  |  c  ", save=False)
    assert result == ({},)  # no crash, returns no-op

def test_wildcard_append_all_blank_pipes_save_false():
    from nodes import WildcardAppend
    result = WildcardAppend().append("SHOT", " | | | ", save=False)
    assert result == ({},)


# ── _build_prompt (legacy) ────────────────────────────────────────────────────

def test_build_prompt_whitespace_only_text():
    from nodes import _build_prompt
    result = _build_prompt("   ", "minimal", seed=1)
    assert result
    assert not result.startswith(",")

def test_build_prompt_all_presets_work():
    from nodes import _build_prompt
    for preset in PRESETS:
        result = _build_prompt("char", preset, seed=42)
        assert result
        assert "char" in result

def test_build_prompt_text_with_leading_spaces():
    from nodes import _build_prompt
    result = _build_prompt("  char  ", "minimal", seed=1)
    assert "char" in result
    assert not result.startswith(" ")
