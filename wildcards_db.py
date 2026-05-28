"""
Wildcard database — shot/scene/pose/lighting/fx categories.
Import this in your generator script.

Add or edit entries freely — one string per line in each list.
Categories are independent; combine them via PRESETS.
"""

import random

# ---------------------------------------------------------------------------
# SHOT TYPES
# ---------------------------------------------------------------------------
WILDCARDS = {}

WILDCARDS["SHOT"] = [
    # Standard framing
    "full body shot, head to toe",
    "cowboy shot, mid-thigh framing",
    "waist up shot, half body",
    "bust shot, chest up portrait",
    "close up portrait, face focus",
    "extreme close up, eyes and lips",
    # Angles
    "dutch angle, tilted frame, dynamic",
    "low angle shot, looking up at character",
    "high angle shot, bird's eye view, looking down",
    "worm's eye view, dramatic upward perspective",
    "over the shoulder shot, pov from behind",
    "profile view, side silhouette",
    "three-quarter view, 3/4 angle",
    "back view, seen from behind",
    # Cinematic
    "wide establishing shot, character in environment",
    "medium shot, waist to head",
    "extreme wide shot, tiny figure in vast landscape",
    "two shot composition, character left of frame",
    "centered symmetrical composition",
    "rule of thirds framing, character offset",
]

# ---------------------------------------------------------------------------
# POSES
# ---------------------------------------------------------------------------
WILDCARDS["POSE"] = [
    # Standing
    "standing, hands on hips, confident pose",
    "standing, arms at sides, relaxed",
    "standing, arms crossed, stern",
    "standing, one hand raised, pointing",
    "standing, arms spread wide, triumphant",
    "leaning against wall, casual",
    "standing on tiptoe, reaching up",
    # Walking / Running
    "walking forward, mid-stride, dynamic",
    "running, full sprint, hair flowing",
    "jogging, relaxed pace",
    "stepping forward, one foot raised",
    # Sitting / Crouching
    "sitting elegantly, legs crossed",
    "sitting on ground, knees to chest",
    "crouching, low to ground, ready to leap",
    "perched on edge, sitting on ledge",
    "kneeling on one knee",
    # Action
    "jumping, airborne, arms raised",
    "mid-air kick, martial arts pose",
    "spinning, dynamic rotation",
    "falling backwards, dramatic",
    "lunging forward, aggressive stance",
    "defensive stance, guard up",
    "casting spell, arms outstretched, magic energy",
    "sword raised overhead, battle pose",
    "drawing bow, archer stance",
    # Relaxed
    "looking over shoulder, glancing back",
    "stretching, arms above head",
    "hands clasped behind back, thoughtful",
    "leaning forward, hands on knees",
    "twirling, dress spinning",
]

# ---------------------------------------------------------------------------
# EXPRESSIONS
# ---------------------------------------------------------------------------
WILDCARDS["EXPRESSION"] = [
    "confident smile, bright eyes",
    "serious determined expression, focused",
    "surprised, wide eyes, open mouth",
    "playful smirk, mischievous look",
    "laughing, head tilted back, joyful",
    "thinking, finger on chin, curious",
    "angry, furrowed brows, fierce",
    "sad, downcast eyes, melancholic",
    "shy, slight blush, averted gaze",
    "fierce battle expression, teeth gritted",
    "mysterious smile, knowing look",
    "excited, sparkling eyes, wide grin",
    "stoic, emotionless, cold gaze",
    "gentle warm smile, soft eyes",
    "shocked, jaw dropped, disbelief",
    "winking, playful",
    "seductive, half-lidded eyes",
]

# ---------------------------------------------------------------------------
# LIGHTING
# ---------------------------------------------------------------------------
WILDCARDS["LIGHTING"] = [
    # Natural
    "soft natural daylight, diffused shadows",
    "golden hour sunlight, warm orange glow",
    "harsh midday sun, strong shadows",
    "overcast soft lighting, no harsh shadows",
    "blue hour, twilight, soft cool light",
    # Practical / Magical
    "candlelight, warm flickering orange glow",
    "torchlight, dramatic dancing shadows",
    "neon lights, colorful reflections",
    "bioluminescent glow, underwater blue-green",
    "lava glow, deep red-orange light",
    "magical aura, soft colored energy light",
    "floating lantern light, warm amber",
    # Cinematic
    "dramatic side lighting, deep shadows, chiaroscuro",
    "backlit, strong rim light, hair glow",
    "three-point studio lighting, clean",
    "moonlight, cool silver-blue shadows",
    "foggy diffused light, atmospheric haze",
    "lightning flash, high contrast dramatic",
    "spotlight, single beam, theatrical",
    "stained glass colored light, kaleidoscopic",
]

# ---------------------------------------------------------------------------
# SCENES / ENVIRONMENTS
# ---------------------------------------------------------------------------
WILDCARDS["SCENE"] = [
    # Fantasy / Magical
    "enchanted garden, floating lanterns, glowing flowers",
    "dark mystical forest, fireflies, ancient trees",
    "ancient magical library, floating books, candlelight",
    "mystical portal, swirling energy vortex",
    "fairy tale castle, stone walls, ivy",
    "floating islands, clouds, waterfalls",
    "underwater ruins, coral, bioluminescent fish",
    "crystal cave, glowing gemstones, reflections",
    "witch's tower, cauldron, arcane symbols",
    "sacred temple ruins, overgrown vines",
    "sky fortress, clouds, aerial view",
    "magical marketplace, glowing stalls, night",
    # Dark / Dramatic
    "volcanic cave, lava rivers, red glow",
    "haunted mansion, broken windows, fog",
    "battlefield ruins, smoke, debris",
    "dark dungeon, stone walls, torch sconces",
    "demonic throne room, hellfire, ominous",
    "apocalyptic cityscape, ruins, red sky",
    # Modern / Urban
    "city rooftop at sunset, skyline panorama",
    "rainy street at night, neon reflections, puddles",
    "grand palace hall, marble columns, chandeliers",
    "modern penthouse, floor to ceiling windows",
    "underground subway, fluorescent lights",
    # Nature
    "snowy mountain peak, aurora borealis, stars",
    "sunny meadow, wildflowers, blue sky, clouds",
    "desert ruins, golden sand dunes, ancient stones",
    "tropical beach, turquoise water, sunset",
    "bamboo forest, misty, filtered light",
    "autumn forest, red-orange leaves, ground fog",
]

# ---------------------------------------------------------------------------
# CROWD / MULTI-PERSON
# ---------------------------------------------------------------------------
WILDCARDS["CROWD"] = [
    "in a crowd of people in background, blurred",
    "surrounded by cheering crowd, stadium",
    "in busy marketplace, people passing by",
    "leading a group of soldiers, army behind",
    "standing apart from crowd, solitary",
    "center of attention, crowd watching",
    "two characters, back to back, allies",
    "facing an opponent, confrontation",
    "group of three, team pose",
    "surrounded by enemies, outnumbered",
    "side by side with another character",
    "character in foreground, silhouettes behind",
    "in a tavern, people at tables, lively atmosphere",
    "at a royal court, nobles watching",
    "street scene, vendors and pedestrians",
]

# ---------------------------------------------------------------------------
# CREATURES / MONSTERS
# ---------------------------------------------------------------------------
WILDCARDS["CREATURE"] = [
    "large dragon in background, wings spread",
    "riding a fantasy horse, galloping",
    "with a wolf companion, loyal beast",
    "confronting a giant spider creature",
    "surrounded by small magical sprites",
    "with a phoenix, fire bird companion",
    "riding a wyvern, aerial mount",
    "facing a hydra, multiple heads",
    "with a giant eagle, majestic bird",
    "facing a demon creature, horned beast",
    "shadow beast emerging from darkness",
    "corrupted knight enemy, dark armor",
    "skeletal undead warriors surrounding",
    "giant tentacled creature rising from water",
    "with a giant wolf mount",
    "in a forest, deer watching nearby",
    "sea monster rising from depths",
    "giant stone golem creature",
]

# ---------------------------------------------------------------------------
# DYNAMIC / ACTION FX
# ---------------------------------------------------------------------------
WILDCARDS["DYNAMIC"] = [
    "casting fire spell, flames swirling",
    "lightning magic, electric bolts crackling",
    "ice magic, frost crystals forming",
    "wind magic, debris and leaves swirling",
    "dark energy eruption, shadow tendrils",
    "healing magic, warm golden light radiating",
    "telekinesis, objects floating around",
    "portal opening, energy vortex spinning",
    "magical shield forming, blocking attack",
    "sword clash, sparks flying",
    "blocking incoming attack, last moment",
    "leaping over explosion, debris",
    "dodging energy beam, quick movement",
    "landing from great height, impact crater",
    "deflecting arrows mid-air",
    "clothes and hair blown by strong wind",
    "standing in heavy rain, soaking wet",
    "surrounded by falling cherry blossoms",
    "debris and dust exploding around",
    "fire spreading in background, dramatic",
    "ground cracking beneath feet",
    "slow motion effect, frozen mid-action",
]

# ---------------------------------------------------------------------------
# WEATHER / ATMOSPHERE
# ---------------------------------------------------------------------------
WILDCARDS["WEATHER"] = [
    "clear sky, bright and vivid",
    "heavy rain, dramatic downpour",
    "light drizzle, soft rain",
    "thick morning fog, misty atmosphere",
    "snowstorm, blizzard conditions",
    "light snow falling gently",
    "stormy sky, dark clouds, lightning",
    "heat haze, shimmering air",
    "smoke and ash in air",
    "aurora borealis, northern lights",
    "sunset with dramatic cloud formation",
    "stars visible, clear night sky",
]

# ---------------------------------------------------------------------------
# TIME OF DAY
# ---------------------------------------------------------------------------
WILDCARDS["TIME"] = [
    "dawn, first light, pale sky",
    "morning, fresh light, gentle shadows",
    "midday, bright harsh light",
    "afternoon, warm golden tones",
    "golden hour, magic hour, sunset",
    "dusk, fading light, purple sky",
    "night, darkness, artificial lights",
    "deep night, moonlight only, starry sky",
]

# ---------------------------------------------------------------------------
# PRESETS — curated category combinations per use case
# ---------------------------------------------------------------------------
PRESETS = {
    # Core dataset — good default for most use cases
    "full":        ["SHOT", "POSE", "EXPRESSION", "LIGHTING", "SCENE"],
    # Action / combat / spells
    "action":      ["SHOT", "POSE", "EXPRESSION", "LIGHTING", "SCENE", "DYNAMIC"],
    # Character in world — environment focus
    "environment": ["SHOT", "POSE", "SCENE", "WEATHER", "TIME"],
    # Social / crowd interactions
    "social":      ["SHOT", "POSE", "EXPRESSION", "SCENE", "CROWD"],
    # Creature encounters
    "creature":    ["SHOT", "POSE", "EXPRESSION", "SCENE", "CREATURE", "LIGHTING"],
    # Maximum variety — all categories
    "chaos":       ["SHOT", "POSE", "EXPRESSION", "LIGHTING", "SCENE", "DYNAMIC", "WEATHER"],
    # Fast / minimal — low artifact risk
    "minimal":     ["SHOT", "POSE", "SCENE"],
}

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def pick(lst, seed=None):
    """Return one random item from a list."""
    if seed is not None:
        random.seed(seed)
    return random.choice(lst)


def build_wildcard_block(preset="full", seed=None):
    """
    Pick one item per category in the preset.

    Returns:
        dict  — { category: chosen_string }
        str   — joined prompt fragment (comma-separated)
        str   — short label safe for filenames
    """
    if seed is not None:
        random.seed(seed)

    picked = {cat: random.choice(WILDCARDS[cat]) for cat in PRESETS[preset]}

    prompt_fragment = ", ".join(picked.values())

    label_parts = []
    for cat in ("SHOT", "POSE"):
        if cat in picked:
            label_parts.append(picked[cat].split(",")[0].strip().replace(" ", "_"))
    label = "_".join(label_parts)[:60]

    return picked, prompt_fragment, label


def stats():
    """Print database statistics."""
    print("=== Wildcard Database ===")
    total = 1
    for cat, items in WILDCARDS.items():
        print(f"  {cat:<12} {len(items):>3} options")
        total *= len(items)
    print(f"\n  Total combinations: {total:,}\n")
    print("  Presets:")
    for name, cats in PRESETS.items():
        n = 1
        for c in cats:
            n *= len(WILDCARDS[c])
        print(f"    {name:<14} {len(cats)} categories → {n:,} combos")


if __name__ == "__main__":
    stats()
