import sys, os, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wildcards_db import PRESETS


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
