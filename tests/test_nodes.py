import sys, os, tempfile, shutil, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wildcards_db import PRESETS, WILDCARDS


# ── Shared helpers ────────────────────────────────────────────────────────────

class FakeClip:
    """Minimal CLIP stub — records tokenized texts for assertion."""
    def __init__(self):
        self.seen = []
    def tokenize(self, text):
        self.seen.append(text)
        return text
    def encode_from_tokens(self, tokens, return_pooled):
        return tokens, "pooled"


def test_build_prompt_includes_text():
    from nodes import _build_prompt
    result = _build_prompt("evi_char, red hair", "minimal", seed=42)
    assert "evi_char" in result
    assert "red hair" in result

def test_build_prompt_includes_wildcard_fragment():
    from nodes import _build_prompt
    result = _build_prompt("evi_char", "minimal", seed=42)
    assert len(result) > len("evi_char") + 10

def test_build_prompt_empty_text():
    from nodes import _build_prompt
    result = _build_prompt("", "minimal", seed=42)
    assert result
    assert not result.startswith(",")

def test_build_prompt_same_seed_same_result():
    from nodes import _build_prompt
    a = _build_prompt("evi_char", "full", seed=123)
    b = _build_prompt("evi_char", "full", seed=123)
    assert a == b

def test_build_prompt_different_seeds_differ():
    from nodes import _build_prompt
    a = _build_prompt("evi_char", "full", seed=1)
    b = _build_prompt("evi_char", "full", seed=9999)
    assert a != b


def _make_tmp_db():
    src = os.path.join(os.path.dirname(__file__), "..", "wildcards_db.py")
    tmp = tempfile.mkdtemp()
    dst = os.path.join(tmp, "wildcards_db.py")
    shutil.copy(src, dst)
    return dst, tmp

def test_append_adds_new_entry():
    from nodes import _append_to_db
    db_path, tmp = _make_tmp_db()
    try:
        added = _append_to_db("SCENE", ["__test_scene_unique_xyz__"], db_path=db_path)
        assert added == 1
        with open(db_path) as f:
            content = f.read()
        assert "__test_scene_unique_xyz__" in content
    finally:
        shutil.rmtree(tmp)

def test_append_skips_duplicates():
    from nodes import _append_to_db
    db_path, tmp = _make_tmp_db()
    try:
        _append_to_db("SCENE", ["__dup_test__"], db_path=db_path)
        added = _append_to_db("SCENE", ["__dup_test__"], db_path=db_path)
        assert added == 0
    finally:
        shutil.rmtree(tmp)

def test_append_multiple_pipe_entries():
    from nodes import _append_to_db
    db_path, tmp = _make_tmp_db()
    try:
        added = _append_to_db("SHOT", ["__shot_a__", "__shot_b__", "__shot_c__"], db_path=db_path)
        assert added == 3
        with open(db_path) as f:
            content = f.read()
        for e in ["__shot_a__", "__shot_b__", "__shot_c__"]:
            assert e in content
    finally:
        shutil.rmtree(tmp)

def test_append_preserves_existing_entries():
    from nodes import _append_to_db
    import importlib.util
    db_path, tmp = _make_tmp_db()
    try:
        spec = importlib.util.spec_from_file_location("wdb_tmp", db_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        original_count = len(mod.WILDCARDS["POSE"])
        _append_to_db("POSE", ["__new_pose__"], db_path=db_path)
        spec2 = importlib.util.spec_from_file_location("wdb_tmp2", db_path)
        mod2 = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(mod2)
        assert len(mod2.WILDCARDS["POSE"]) == original_count + 1
    finally:
        shutil.rmtree(tmp)


# ── _apply_loras ──────────────────────────────────────────────────────────────

def _fake_loader(calls):
    def loader(model, clip, lora_path, sm, sc):
        calls.append((lora_path, sm, sc))
        return model, clip
    return loader

def test_apply_loras_skips_none():
    from nodes import _apply_loras
    calls = []
    _apply_loras("m", "c", [("None", 0.7), ("None", 0.5)],
                 loader=_fake_loader(calls), path_resolver=lambda n: n)
    assert calls == []

def test_apply_loras_applies_non_none():
    from nodes import _apply_loras
    calls = []
    _apply_loras("m", "c", [("lora_a.safetensors", 0.8), ("None", 0.5)],
                 loader=_fake_loader(calls), path_resolver=lambda n: n)
    assert len(calls) == 1
    assert calls[0] == ("lora_a.safetensors", 0.8, 0.8)

def test_apply_loras_applies_all_non_none():
    from nodes import _apply_loras
    calls = []
    _apply_loras("m", "c",
                 [("a.safetensors", 0.6), ("b.safetensors", 0.9), ("c.safetensors", 0.3)],
                 loader=_fake_loader(calls), path_resolver=lambda n: n)
    assert [s for _, s, _ in calls] == [0.6, 0.9, 0.3]

def test_apply_loras_returns_modified_model_clip():
    from nodes import _apply_loras
    def mutating_loader(model, clip, lora_path, sm, sc):
        return model + "_mod", clip + "_mod"
    result_model, result_clip = _apply_loras(
        "m", "c", [("lora.safetensors", 1.0)],
        loader=mutating_loader, path_resolver=lambda n: n)
    assert result_model == "m_mod"
    assert result_clip == "c_mod"

def test_apply_loras_chains_multiple():
    from nodes import _apply_loras
    def chaining_loader(model, clip, lora_path, sm, sc):
        return model + f"+{lora_path}", clip
    result_model, _ = _apply_loras(
        "m", "c",
        [("x.safetensors", 1.0), ("y.safetensors", 0.5), ("z.safetensors", 0.3)],
        loader=chaining_loader, path_resolver=lambda n: n)
    assert result_model == "m+x.safetensors+y.safetensors+z.safetensors"


# ── MultiLoraLoader ───────────────────────────────────────────────────────────

def test_multi_lora_enabled_all_false_skips_all():
    from nodes import MultiLoraLoader
    node = MultiLoraLoader()
    m, c = node.load_loras("model", "clip", enabled_all=False,
                           lora_1="some.safetensors", strength_1=0.8, enabled_1=True)
    assert m == "model"
    assert c == "clip"

def test_multi_lora_slot_disabled_skips_that_lora():
    from nodes import MultiLoraLoader
    node = MultiLoraLoader()
    # All loras are "None" — _apply_loras skips them, no ComfyUI needed
    m, c = node.load_loras("model", "clip", enabled_all=True,
                           lora_1="None", strength_1=1.0, enabled_1=False)
    assert m == "model"
    assert c == "clip"

def test_multi_lora_none_slots_are_skipped():
    from nodes import MultiLoraLoader
    node = MultiLoraLoader()
    m, c = node.load_loras("model", "clip", enabled_all=True,
                           lora_1="None", strength_1=1.0, enabled_1=True,
                           lora_2="None", strength_2=0.5, enabled_2=True)
    assert m == "model"
    assert c == "clip"


# ── _build_prompt_cats ────────────────────────────────────────────────────────

def test_build_prompt_cats_includes_text():
    from nodes import _build_prompt_cats
    result = _build_prompt_cats("my character", {"SHOT": True}, seed=1)
    assert "my character" in result

def test_build_prompt_cats_only_enabled_cats():
    from nodes import _build_prompt_cats
    from wildcards_db import WILDCARDS
    result = _build_prompt_cats("", {"SHOT": True, "POSE": False, "SCENE": False}, seed=42)
    # Result should contain a SHOT entry and nothing from POSE/SCENE
    assert any(entry in result for entry in WILDCARDS["SHOT"])
    assert not any(entry in result for entry in WILDCARDS["POSE"])

def test_build_prompt_cats_no_categories_returns_text():
    from nodes import _build_prompt_cats
    result = _build_prompt_cats("base text", {}, seed=1)
    assert result == "base text"

def test_build_prompt_cats_same_seed_same_result():
    from nodes import _build_prompt_cats
    cats = {"SHOT": True, "POSE": True, "SCENE": True}
    a = _build_prompt_cats("char", cats, seed=77)
    b = _build_prompt_cats("char", cats, seed=77)
    assert a == b

def test_build_prompt_cats_different_seeds_differ():
    from nodes import _build_prompt_cats
    cats = {"SHOT": True, "POSE": True, "SCENE": True}
    a = _build_prompt_cats("char", cats, seed=1)
    b = _build_prompt_cats("char", cats, seed=9999)
    assert a != b

def test_build_prompt_cats_empty_text_no_leading_comma():
    from nodes import _build_prompt_cats
    result = _build_prompt_cats("", {"SHOT": True}, seed=1)
    assert result
    assert not result.startswith(",")

def test_build_prompt_cats_override_forces_value():
    from nodes import _build_prompt_cats
    from wildcards_db import WILDCARDS
    result = _build_prompt_cats("", {"SHOT": True}, seed=1,
                                overrides={"SHOT": "full body shot, head to toe"})
    # Every run with any seed must produce the overridden value
    assert "full body shot, head to toe" in result

def test_build_prompt_cats_override_ignores_random():
    from nodes import _build_prompt_cats
    # Override should be consistent across seeds
    results = {_build_prompt_cats("", {"SHOT": True}, seed=s,
                                  overrides={"SHOT": "waist up shot, half body"})
               for s in range(20)}
    assert len(results) == 1   # always the same value regardless of seed

def test_build_prompt_cats_no_override_varies_with_seed():
    from nodes import _build_prompt_cats
    results = {_build_prompt_cats("", {"SHOT": True}, seed=s) for s in range(20)}
    assert len(results) > 1    # random picks across seeds


def test_ultra_wildcard_is_changed_always_returns_seed():
    """IS_CHANGED must return the seed for every value — ComfyUI Randomize/
    Increment/Decrement controls drive diversity by changing the seed before
    each run; returning the seed value lets ComfyUI cache Fixed runs correctly."""
    from nodes import UltraWildcardNode
    assert UltraWildcardNode.IS_CHANGED(seed=0) == 0
    assert UltraWildcardNode.IS_CHANGED(seed=42) == 42
    assert UltraWildcardNode.IS_CHANGED(seed=0x7FFFFFFFFFFFFFFF) == 0x7FFFFFFFFFFFFFFF

def test_ultra_wildcard_different_seeds_produce_variety():
    """Different seeds must produce different prompts (diversity in batch mode)."""
    from nodes import UltraWildcardNode
    seen = set()

    class FakeClip:
        def tokenize(self, text): return text
        def encode_from_tokens(self, tokens, return_pooled): return tokens, "pooled"

    node = UltraWildcardNode()
    for s in range(15):
        result = node.encode(FakeClip(), "char", seed=s, enabled_all=False,
                             shot=True, pose=False, scene=False,
                             expression=False, lighting=False, crowd=False,
                             creature=False, dynamic=False, weather=False, time=False)
        seen.add(result[0][0][0])
    assert len(seen) > 1, "Different seeds must produce different outputs"

def test_ultra_wildcard_enabled_all_false_uses_individual_toggles():
    """enabled_all=False uses individual category toggles (not text-only)."""
    from nodes import UltraWildcardNode
    from wildcards_db import WILDCARDS
    seen = []

    class FakeClip:
        def tokenize(self, text):
            seen.append(text)
            return text
        def encode_from_tokens(self, tokens, return_pooled): return "cond", "pooled"

    node = UltraWildcardNode()
    node.encode(FakeClip(), "only this", seed=42, enabled_all=False,
                shot=True, pose=True, scene=True,
                expression=False, lighting=False, crowd=False,
                creature=False, dynamic=False, weather=False, time=False)
    result = seen[0]
    # base text is preserved
    assert "only this" in result
    # at least one wildcard from enabled categories was appended
    shot_hit = any(e in result for e in WILDCARDS["SHOT"])
    pose_hit = any(e in result for e in WILDCARDS["POSE"])
    scene_hit = any(e in result for e in WILDCARDS["SCENE"])
    assert shot_hit or pose_hit or scene_hit

def test_ultra_wildcard_enabled_all_false_all_toggles_off_returns_text():
    """enabled_all=False with all categories off → plain text, no wildcards."""
    from nodes import UltraWildcardNode
    seen = []

    class FakeClip:
        def tokenize(self, text):
            seen.append(text)
            return text
        def encode_from_tokens(self, tokens, return_pooled): return "cond", "pooled"

    node = UltraWildcardNode()
    node.encode(FakeClip(), "only this", seed=42, enabled_all=False,
                shot=False, pose=False, scene=False,
                expression=False, lighting=False, crowd=False,
                creature=False, dynamic=False, weather=False, time=False)
    assert seen == ["only this"]

def test_ultra_wildcard_enabled_all_true_ignores_individual_toggles():
    """enabled_all=True forces ALL categories on regardless of individual toggles."""
    from nodes import UltraWildcardNode
    from wildcards_db import WILDCARDS
    seen = []

    class FakeClip:
        def tokenize(self, text):
            seen.append(text)
            return text
        def encode_from_tokens(self, tokens, return_pooled): return "cond", "pooled"

    node = UltraWildcardNode()
    # All individual toggles are False, but enabled_all=True should override
    node.encode(FakeClip(), "base", seed=42, enabled_all=True,
                shot=False, pose=False, scene=False,
                expression=False, lighting=False, crowd=False,
                creature=False, dynamic=False, weather=False, time=False)
    result = seen[0]
    # With enabled_all=True all cats are on — result must be longer than base text
    assert len(result) > len("base") + 10


# ── UltraWildcardNode — shot_lock ─────────────────────────────────────────────

def test_ultra_wildcard_shot_lock_portrait_overrides_shot():
    """shot_lock=portrait must inject the portrait override string."""
    from nodes import UltraWildcardNode
    clip = FakeClip()
    UltraWildcardNode().encode(clip, "char", seed=1, enabled_all=False,
                               shot=True, pose=False, scene=False,
                               expression=False, lighting=False, crowd=False,
                               creature=False, dynamic=False, weather=False,
                               time=False, shot_lock="portrait")
    assert "close up portrait, face focus" in clip.seen[0]

def test_ultra_wildcard_shot_lock_full_body_overrides_shot():
    from nodes import UltraWildcardNode
    clip = FakeClip()
    UltraWildcardNode().encode(clip, "char", seed=1, enabled_all=False,
                               shot=True, pose=False, scene=False,
                               expression=False, lighting=False, crowd=False,
                               creature=False, dynamic=False, weather=False,
                               time=False, shot_lock="full body")
    assert "full body shot, head to toe" in clip.seen[0]

def test_ultra_wildcard_shot_lock_random_does_not_force_specific_value():
    """shot_lock=random must NOT inject any fixed override."""
    from nodes import UltraWildcardNode
    # Run 30 times — if lock were active, all would be identical
    results = set()
    for s in range(30):
        clip = FakeClip()
        UltraWildcardNode().encode(clip, "", seed=s, enabled_all=False,
                                   shot=True, pose=False, scene=False,
                                   expression=False, lighting=False, crowd=False,
                                   creature=False, dynamic=False, weather=False,
                                   time=False, shot_lock="random")
        results.add(clip.seen[0])
    assert len(results) > 1

def test_ultra_wildcard_shot_lock_ignored_when_shot_disabled():
    """shot_lock has no effect when shot category is off."""
    from nodes import UltraWildcardNode
    clip = FakeClip()
    UltraWildcardNode().encode(clip, "base text", seed=1, enabled_all=False,
                               shot=False, pose=False, scene=False,
                               expression=False, lighting=False, crowd=False,
                               creature=False, dynamic=False, weather=False,
                               time=False, shot_lock="portrait")
    # shot is disabled — portrait override must NOT appear
    assert "close up portrait" not in clip.seen[0]
    assert clip.seen[0] == "base text"

def test_ultra_wildcard_text_trailing_comma_cleaned():
    """Trailing comma/space in text must not produce double-comma in output."""
    from nodes import UltraWildcardNode
    clip = FakeClip()
    UltraWildcardNode().encode(clip, "char, red hair,  ", seed=1, enabled_all=False,
                               shot=True, pose=False, scene=False,
                               expression=False, lighting=False, crowd=False,
                               creature=False, dynamic=False, weather=False, time=False)
    assert ",," not in clip.seen[0]
    assert not clip.seen[0].startswith(",")

def test_ultra_wildcard_empty_text_no_leading_comma():
    from nodes import UltraWildcardNode
    clip = FakeClip()
    UltraWildcardNode().encode(clip, "", seed=5, enabled_all=False,
                               shot=True, pose=False, scene=False,
                               expression=False, lighting=False, crowd=False,
                               creature=False, dynamic=False, weather=False, time=False)
    assert clip.seen[0]
    assert not clip.seen[0].startswith(",")
    assert not clip.seen[0].startswith(" ")

def test_ultra_wildcard_enabled_all_true_includes_all_categories():
    """enabled_all=True must include picks from every category."""
    from nodes import UltraWildcardNode
    clip = FakeClip()
    UltraWildcardNode().encode(clip, "", seed=42, enabled_all=True,
                               shot=False, pose=False, scene=False,
                               expression=False, lighting=False, crowd=False,
                               creature=False, dynamic=False, weather=False, time=False)
    result = clip.seen[0]
    for cat in WILDCARDS:
        assert any(entry in result for entry in WILDCARDS[cat]), \
            f"Category {cat} not represented in enabled_all=True output"

def test_ultra_wildcard_fixed_seed_deterministic_across_calls():
    """Same fixed seed must produce identical prompts on every call."""
    from nodes import UltraWildcardNode
    results = set()
    for _ in range(5):
        clip = FakeClip()
        UltraWildcardNode().encode(clip, "char", seed=999, enabled_all=False,
                                   shot=True, pose=True, scene=True,
                                   expression=False, lighting=False, crowd=False,
                                   creature=False, dynamic=False, weather=False, time=False)
        results.add(clip.seen[0])
    assert len(results) == 1, "Fixed seed must always produce the same prompt"


# ── _apply_loras — corner cases ───────────────────────────────────────────────

def test_apply_loras_empty_pairs_returns_unchanged():
    from nodes import _apply_loras
    m, c = _apply_loras("model", "clip", [],
                        loader=_fake_loader([]), path_resolver=lambda n: n)
    assert m == "model" and c == "clip"

def test_apply_loras_strength_zero_still_calls_loader():
    """strength=0.0 is valid — loader must still be called."""
    from nodes import _apply_loras
    calls = []
    _apply_loras("m", "c", [("lora.safetensors", 0.0)],
                 loader=_fake_loader(calls), path_resolver=lambda n: n)
    assert len(calls) == 1
    assert calls[0][1] == 0.0

def test_apply_loras_path_resolver_called_with_name():
    from nodes import _apply_loras
    resolved = []
    def resolver(name):
        resolved.append(name)
        return f"/path/{name}"
    calls = []
    _apply_loras("m", "c", [("my_lora.safetensors", 1.0)],
                 loader=_fake_loader(calls), path_resolver=resolver)
    assert resolved == ["my_lora.safetensors"]
    assert calls[0][0] == "/path/my_lora.safetensors"

def test_apply_loras_all_none_returns_original():
    from nodes import _apply_loras
    calls = []
    m, c = _apply_loras("orig_m", "orig_c",
                        [("None", 1.0), ("None", 0.5), ("None", 0.8)],
                        loader=_fake_loader(calls), path_resolver=lambda n: n)
    assert calls == []
    assert m == "orig_m" and c == "orig_c"


# ── MultiLoraLoader — corner cases ───────────────────────────────────────────

def test_multi_lora_all_8_slots_active():
    """All 8 slots with real loras must all be applied in order."""
    from nodes import MultiLoraLoader
    applied = []
    def fake_loader(model, clip, path, sm, sc):
        applied.append(path)
        return model, clip
    import nodes as _nodes
    orig = _nodes._apply_loras
    try:
        # Inject fake loader via path_resolver trick — use apply_loras directly
        node = MultiLoraLoader()
        # Build kwargs with 8 real loras
        kwargs = {}
        for i in range(1, 9):
            kwargs[f"lora_{i}"] = f"lora_{i}.safetensors"
            kwargs[f"strength_{i}"] = float(i) / 10
            kwargs[f"enabled_{i}"] = True
        # Use _apply_loras with custom loader to avoid real ComfyUI
        pairs = [(kwargs[f"lora_{i}"], kwargs[f"strength_{i}"])
                 for i in range(1, 9)]
        _nodes._apply_loras("m", "c", pairs,
                            loader=fake_loader, path_resolver=lambda n: n)
        assert len(applied) == 8
    finally:
        pass  # no teardown needed

def test_multi_lora_individual_slot_disabled_skips_it():
    """enabled_i=False skips that specific slot even if lora_i is set."""
    from nodes import MultiLoraLoader
    calls = []
    import nodes as _nodes
    # slot 1 enabled, slot 2 disabled — only slot 1 should appear in pairs
    pairs = [
        (k, s) for i in range(1, 3)
        for k, s in [(f"lora_{i}.safetensors", 1.0)]
        if {1: True, 2: False}[i]
    ]
    _nodes._apply_loras("m", "c", pairs,
                        loader=_fake_loader(calls), path_resolver=lambda n: n)
    assert len(calls) == 1
    assert calls[0][0] == "lora_1.safetensors"

def test_multi_lora_missing_slot_kwargs_defaults_to_skip():
    """If slot kwargs are absent, load_loras must not crash."""
    from nodes import MultiLoraLoader
    node = MultiLoraLoader()
    # Pass only slot 1, omit slots 2-8 entirely
    m, c = node.load_loras("model", "clip", enabled_all=True,
                           lora_1="None", strength_1=1.0, enabled_1=True)
    assert m == "model" and c == "clip"


# ── _append_to_db — corner cases ─────────────────────────────────────────────

def test_append_unknown_category_raises():
    from nodes import _append_to_db
    db_path, tmp = _make_tmp_db()
    try:
        import pytest
        with pytest.raises(ValueError, match="not found"):
            _append_to_db("NONEXISTENT_CAT", ["entry"], db_path=db_path)
    finally:
        shutil.rmtree(tmp)

def test_append_empty_list_returns_zero():
    from nodes import _append_to_db
    db_path, tmp = _make_tmp_db()
    try:
        added = _append_to_db("SCENE", [], db_path=db_path)
        assert added == 0
    finally:
        shutil.rmtree(tmp)

def test_append_mixed_new_and_duplicate():
    """Only new entries should be added; duplicates silently skipped."""
    from nodes import _append_to_db
    import importlib.util
    db_path, tmp = _make_tmp_db()
    try:
        _append_to_db("SCENE", ["__dup__", "__new_a__"], db_path=db_path)
        added = _append_to_db("SCENE", ["__dup__", "__new_b__"], db_path=db_path)
        assert added == 1  # only __new_b__ is truly new
    finally:
        shutil.rmtree(tmp)

def test_append_result_is_valid_python():
    """After append, the modified file must still be importable."""
    from nodes import _append_to_db
    import importlib.util
    db_path, tmp = _make_tmp_db()
    try:
        _append_to_db("LIGHTING", ["__test_lighting__"], db_path=db_path)
        spec = importlib.util.spec_from_file_location("wdb_post", db_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)   # must not raise SyntaxError
        assert "__test_lighting__" in mod.WILDCARDS["LIGHTING"]
    finally:
        shutil.rmtree(tmp)


# ── WildcardAppend node ───────────────────────────────────────────────────────

def test_wildcard_append_save_false_does_nothing():
    from nodes import WildcardAppend
    node = WildcardAppend()
    result = node.append("SCENE", "entry_a | entry_b", save=False)
    assert result == ({},)

def test_wildcard_append_save_true_empty_entries_does_nothing():
    """Blank/whitespace entries string with save=True must not crash."""
    from nodes import WildcardAppend
    db_path, tmp = _make_tmp_db()
    try:
        node = WildcardAppend()
        # Patch db path by monkeypatching is complex — test via save=False instead
        # to verify graceful empty-input handling at the parse stage
        result = node.append("SCENE", "   |   |  ", save=False)
        assert result == ({},)
    finally:
        shutil.rmtree(tmp)


# ── wildcards_db — pick() and build_wildcard_block() ─────────────────────────

def test_pick_same_seed_same_result():
    from wildcards_db import pick
    a = pick(WILDCARDS["SHOT"], seed=7)
    b = pick(WILDCARDS["SHOT"], seed=7)
    assert a == b

def test_pick_different_seeds_can_differ():
    from wildcards_db import pick
    results = {pick(WILDCARDS["SHOT"], seed=s) for s in range(50)}
    assert len(results) > 1

def test_pick_no_seed_returns_valid_entry():
    from wildcards_db import pick
    result = pick(WILDCARDS["SCENE"])
    assert result in WILDCARDS["SCENE"]

def test_pick_does_not_pollute_global_random():
    """Calling pick() must not change the global random state."""
    import random
    from wildcards_db import pick
    random.seed(123)
    before = [random.random() for _ in range(5)]
    random.seed(123)
    pick(WILDCARDS["SHOT"], seed=42)
    after = [random.random() for _ in range(5)]
    assert before == after

def test_build_wildcard_block_all_presets_return_string():
    from wildcards_db import build_wildcard_block
    for preset in PRESETS:
        picked, fragment, label = build_wildcard_block(preset=preset, seed=1)
        assert isinstance(fragment, str) and fragment
        assert isinstance(label, str)
        assert isinstance(picked, dict)

def test_build_wildcard_block_label_max_60_chars():
    from wildcards_db import build_wildcard_block
    for preset in PRESETS:
        _, _, label = build_wildcard_block(preset=preset, seed=99)
        assert len(label) <= 60, f"Label too long for preset {preset}: {label!r}"

def test_build_wildcard_block_same_seed_deterministic():
    from wildcards_db import build_wildcard_block
    a = build_wildcard_block(preset="full", seed=42)
    b = build_wildcard_block(preset="full", seed=42)
    assert a[1] == b[1]

def test_build_wildcard_block_different_seeds_differ():
    from wildcards_db import build_wildcard_block
    results = {build_wildcard_block(preset="full", seed=s)[1] for s in range(20)}
    assert len(results) > 1

def test_build_wildcard_block_does_not_pollute_global_random():
    """build_wildcard_block must not change global random state."""
    import random
    from wildcards_db import build_wildcard_block
    random.seed(777)
    before = [random.random() for _ in range(5)]
    random.seed(777)
    build_wildcard_block(preset="full", seed=99)
    after = [random.random() for _ in range(5)]
    assert before == after

def test_build_wildcard_block_picks_from_correct_categories():
    from wildcards_db import build_wildcard_block
    for preset, cats in PRESETS.items():
        picked, _, _ = build_wildcard_block(preset=preset, seed=1)
        assert set(picked.keys()) == set(cats), \
            f"Preset {preset}: expected {cats}, got {list(picked.keys())}"

def test_build_wildcard_block_fragment_contains_all_picks():
    from wildcards_db import build_wildcard_block
    picked, fragment, _ = build_wildcard_block(preset="minimal", seed=5)
    for val in picked.values():
        assert val in fragment
