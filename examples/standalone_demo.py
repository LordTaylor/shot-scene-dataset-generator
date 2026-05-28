"""
Standalone demo — no ComfyUI required.
Shows how to use wildcards_db without any server.

Run:
    python3 examples/standalone_demo.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from wildcards_db import build_wildcard_block, pick, WILDCARDS, stats

print("=" * 60)
print("  WILDCARD DATABASE — standalone demo")
print("=" * 60)

stats()

print("\n" + "=" * 60)
print("  SAMPLE PROMPT BLOCKS")
print("=" * 60)

for preset_name in ["minimal", "full", "action", "chaos"]:
    picked, fragment, label = build_wildcard_block(preset=preset_name, seed=42)
    print(f"\n[Preset: {preset_name}]")
    print(f"  Label:    {label}")
    for cat, val in picked.items():
        print(f"  {cat:<12} {val}")
    print(f"  Prompt:   {fragment[:100]}...")

print("\n" + "=" * 60)
print("  RANDOM PICKS FROM INDIVIDUAL CATEGORIES")
print("=" * 60)

for cat in WILDCARDS:
    sample = pick(WILDCARDS[cat])
    print(f"  {cat:<12} {sample}")
