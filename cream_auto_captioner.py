"""
Cream Auto Captioner Node for ComfyUI

Automatically tags images in a dataset folder using WD tagger models (ONNX).
Optionally filters appearance/body tags for character LoRA training.

Based on pythongosssss/ComfyUI-WD14-Tagger (MIT License).
Tag filtering logic based on danbooru_tag_pruner.py.
"""

import os
import re
import csv
import numpy as np

from PIL import Image

try:
    import comfy.utils
except Exception:
    comfy = None

# ── Constants ──────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')

# Models available for download from HuggingFace
KNOWN_MODELS = {
    "wd-eva02-large-tagger-v3": "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3",
    "wd-vit-tagger-v3": "https://huggingface.co/SmilingWolf/wd-vit-tagger-v3",
    "wd-swinv2-tagger-v3": "https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3",
    "wd-convnext-tagger-v3": "https://huggingface.co/SmilingWolf/wd-convnext-tagger-v3",
    "wd-v1-4-moat-tagger-v2": "https://huggingface.co/SmilingWolf/wd-v1-4-moat-tagger-v2",
    "wd-v1-4-convnextv2-tagger-v2": "https://huggingface.co/SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
    "wd-v1-4-convnext-tagger-v2": "https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger-v2",
    "wd-v1-4-vit-tagger-v2": "https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger-v2",
    "wd-v1-4-swinv2-tagger-v2": "https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2",
}

# Local directory for storing downloaded models
MODELS_DIR = os.path.join(os.path.dirname(__file__), "tagger_models")


# ══════════════════════════════════════════════════════════════════════════
# Tag Filter — Vocabulary & Patterns
# Removes fixed appearance tags so the LoRA learns them from the images.
# Keeps variable attributes (expressions, poses, composition, backgrounds).
# ══════════════════════════════════════════════════════════════════════════

_COLORS = (
    "red", "orange", "yellow", "green", "blue", "purple", "pink",
    "brown", "black", "white", "grey", "gray", "silver", "gold", "golden",
    "blonde", "blond", "cyan", "teal", "violet", "indigo", "magenta",
    "crimson", "dark", "light", "pale", "bright", "multicolored",
    "multi-colored", "gradient", "streaked", "two-tone",
    "aqua", "lavender", "lilac", "maroon", "navy", "olive",
    "coral", "turquoise", "platinum", "copper", "auburn",
    "amber", "hazel", "emerald", "sapphire", "ruby", "scarlet",
    "ochre", "tan", "cream", "ivory", "charcoal", "ash",
)
_CP = "(?:" + "|".join(_COLORS) + ")"

_SIZE_MODIFIERS = (
    "large", "huge", "big", "massive", "giant", "enormous",
    "gigantic", "oversized",
    "small", "tiny", "mini", "flat",
    "thick", "wide", "broad",
    "narrow", "slim", "thin",
    "round", "plump", "puffy", "perky", "pointed", "soft",
    "long", "short", "tall",
    "heavy", "toned", "defined",
)
_SM = "(?:" + "|".join(_SIZE_MODIFIERS) + ")"

_BODY_PARTS = (
    "pectorals?", "chest", "breasts?", "bust", "nipples?",
    "biceps?", "arms?", "muscles?", "shoulders?", "back",
    "abs?", "abdomen", "stomach", "belly", "waist", "torso",
    "thighs?", "hips?", "buttocks?", "butt", "ass", "legs?",
    "calves?", "ankles?", "feet", "foot",
    "face", "cheeks?", "lips?", "nose", "jaw", "chin", "forehead",
    "body", "build", "figure", "frame", "stature",
    "paws?", "claws?", "muzzle", "snout",
)
_BP = "(?:" + "|".join(_BODY_PARTS) + ")"

# Regex patterns — each (category_name, pattern)
_FILTER_PATTERNS_RAW = [
    # ── Eyes ──
    ("eye color",
     rf"^{_CP}[\s_]eyes?$"),
    ("eye - iris/pupil/sclera",
     rf"^{_CP}[\s_](?:iris|pupil|sclera)s?$"),
    ("eye - heterochromia",
     r"^heterochromia"),
    ("eye - shape/special",
     r"^(?:tsurime|tareme|sanpaku|jitome|"
     r"half[\s_]closed[\s_]eyes?|empty[\s_]eyes?|blank[\s_]eyes?|"
     r"glowing[\s_]eyes?|slit[\s_]pupils?|vertical[\s_]pupils?|"
     r"star[\s_](?:shaped[\s_])?pupils?|heart[\s_]pupils?|"
     r"cross[\s_]pupils?|spiral[\s_]pupils?|"
     r"ringed[\s_]eyes?|unusual[\s_]pupils?)$"),

    # ── Hair ──
    ("hair color (compound)",
     rf"^(?:"
     rf"(?:(?:very[\s_])?(?:short|medium|long|very[\s_]long)[\s_]+)?(?:{_CP}[\s_]*)+"
     rf"hair"
     rf"|(?:{_CP}[\s_]*)+(?:very[\s_])?(?:short|medium|long|very[\s_]long)[\s_]+hair"
     rf")$"),
    ("hair tip color",
     rf"^{_CP}[\s_]hair[\s_]tips?$"),
    ("hair - inner/outer color",
     rf"^{_CP}[\s_](?:inner|outer)[\s_]hair$"),
    ("hair length",
     r"^(?:very[\s_])?(?:short|medium|long|very[\s_]long)[\s_]hair$"),
    ("hairstyle - tied",
     r"^(?:twintails?|twin[\s_]tails?|ponytail|low[\s_](?:twin[\s_])?tails?|"
     r"side[\s_](?:tail|ponytail)|pigtails?|braids?|single[\s_]braid|"
     r"double[\s_]bun|hair[\s_]bun|bun[\s_]hair|two[\s_]side[\s_]up|"
     r"one[\s_]side[\s_]up|high[\s_]ponytail|low[\s_]ponytail|"
     r"half[\s_]updo|updo|swept[\s_]hair|crown[\s_]braid)s?$"),
    ("hairstyle - other",
     r"^(?:ahoge|bob[\s_]cut|hime[\s_]cut|bowl[\s_]cut|"
     r"hair[\s_]over[\s_]one[\s_]eye|hair[\s_]over[\s_]eyes?|"
     r"hair[\s_]between[\s_]eyes|asymmetrical[\s_]hair|"
     r"messy[\s_]hair|wavy[\s_]hair|curly[\s_]hair|straight[\s_]hair|"
     r"flipped[\s_]hair|bangs?|blunt[\s_]bangs?|swept[\s_]bangs?|"
     r"parted[\s_]bangs?|hair[\s_]spread[\s_]out|floating[\s_]hair|"
     r"short[\s_]bangs?|sidelocks?|side[\s_]locks?|"
     r"drill[\s_](?:hair|sidelocks?)|ringlets?)$"),
    ("hair accessory",
     r"^(?:hair[\s_]ornament|hair[\s_]flower|hair[\s_]ribbon|"
     r"hair[\s_]bow|hair[\s_]clip|hair[\s_]stick|hair[\s_]pin|"
     r"hair[\s_]tie|hair[\s_]scrunchie|scrunchie|kanzashi|"
     r"hair[\s_]antenna|antenna[\s_]hair|hair[\s_]intakes?)$"),

    # ── Skin ──
    ("skin color",
     rf"^{_CP}[\s_]skin$"),
    ("skin - other",
     r"^(?:tan|tanned|dark[\s_]skin|light[\s_]skin|pale[\s_]skin|"
     r"fair[\s_]skin|doll[\s_]joints?)$"),

    # ── Body: size modifier + part (Layer 1) ──
    ("body - modifier+part",
     rf"^{_SM}[\s_]{_BP}$"),
    ("body - part+modifier (reversed)",
     rf"^{_BP}[\s_]{_SM}$"),
    ("body - intensifier compound",
     r"^(?:very|extremely|incredibly|absurdly|ridiculously|"
     r"super|hyper|mega|ultra)[\s_]"
     rf"{_SM}[\s_]{_BP}$"),
    ("body - gendered size",
     r"^(?:large|huge|tall|short|small|big|tiny|"
     r"massive|giant|enormous|gigantic)[\s_](?:male|female|man|woman)$"),

    # ── Body type (Layer 2) ──
    ("body type - muscular",
     r"^(?:muscular|athletic|well[\s_]built|buff|beefy|"
     r"ripped|jacked|swole|"
     r"bara|muscle[\s_](?:girl|boy|woman)|"
     r"muscular[\s_](?:male|female|arms?|legs?|thighs?|abs?|build))$"),
    ("body type - heavy",
     r"^(?:chubby|overweight|obese|fat|pudgy|bbw|"
     r"chubby[\s_](?:male|female)|weight[\s_]gain)$"),
    ("body type - curvy/slim",
     r"^(?:curvy|voluptuous|hourglass[\s_]figure|"
     r"slender|slim|lanky|willowy|skinny|lean|"
     r"petite|petite[\s_](?:girl|female|body)|"
     r"small[\s_]stature|short[\s_]stature|tall[\s_]stature)$"),
    ("body type - age feel",
     r"^(?:mature|mature[\s_](?:male|female|woman|man)|"
     r"aged|elderly|middle[\s_]aged|"
     r"young[\s_](?:male|female)|"
     r"loli|shota|legal[\s_]loli)$"),
    ("body type - furry specific",
     r"^(?:stocky|heavyset|thick[\s_](?:body|fur)|fluffy[\s_]body)$"),

    # ── Non-human body ──
    ("animal ears",
     r"^(?:cat|dog|fox|wolf|rabbit|bunny|hare|bear|panda|"
     r"raccoon|tanuki|deer|horse|mouse|rat|sheep|cow|pig|"
     r"lion|tiger|dragon|demon|devil|oni|yokai|"
     r"squirrel|bat|otter|ferret|hedgehog|"
     r"bird|eagle|owl|hawk)[\s_]ears?$"),
    ("animal ears - generic",
     r"^(?:animal[\s_]ears?|kemonomimi|pointy[\s_]ears?|"
     r"elf[\s_]ears?|long[\s_]ears?)$"),
    ("tail",
     r"^(?:cat|dog|fox|wolf|rabbit|bunny|bear|horse|deer|"
     r"mouse|rat|sheep|cow|pig|raccoon|tanuki|tiger|lion|"
     r"dragon|demon|devil|oni|squirrel|bat|otter|"
     r"fluffy|multiple|nine[\s_]?tailed?)[\s_]tails?$"),
    ("wings",
     r"^(?:angel|bat|demon|dragon|fairy|butterfly|bird|"
     r"feathered|mechanical|mecha|crystal|insect|"
     r"small|large|huge|giant|tiny)[\s_]wings?$"),
    ("horns",
     r"^(?:horns?|demon[\s_]horns?|dragon[\s_]horns?|"
     r"oni[\s_]horns?|single[\s_]horn|multiple[\s_]horns?|"
     r"curved[\s_]horns?|straight[\s_]horns?|"
     r"short[\s_]horns?|long[\s_]horns?)$"),
    ("special body markings",
     r"^(?:pointy[\s_]nose|button[\s_]nose|"
     r"freckles?|moles?|beauty[\s_]mark|"
     r"scars?|scar[\s_](?:on|across|over|under|beneath|through)[\s_]\w+|"
     r"eyepatch|mechanical[\s_]eye|extra[\s_]eyes?|cyclops|"
     r"body[\s_]markings?|tattoos?|gills?|shark[\s_]teeth|third[\s_]eye)$"),

    # ── Species ──
    ("species - fantasy",
     r"^(?:elf|half[\s_]elf|dark[\s_]elf|high[\s_]elf|wood[\s_]elf|"
     r"vampire|werewolf|dhampir|"
     r"android|robot|cyborg|mecha[\s_](?:girl|boy)|"
     r"demon|devil|angel|fallen[\s_]angel|"
     r"lamia|centaur|harpy|"
     r"ghost|spirit|undead|zombie|skeleton|lich|"
     r"oni|yokai|kitsune|tanuki|tengu|"
     r"fairy|pixie|sprite|succubus|incubus|"
     r"alien|monster[\s_](?:girl|boy)|slime[\s_]girl|"
     r"dragon(?:girl|boy|maid)?|dragonkin|"
     r"orc|goblin|kobold|gnome|dwarf)$"),
    ("species - kemono hybrid",
     r"^(?:cat|dog|fox|wolf|rabbit|bunny|"
     r"cow|horse|bear|sheep|deer|dragon)"
     r"[\s_](?:girl|boy|woman|man|maid)$"),

    # ── Furry/Kemono specific ──
    ("furry - species+gender",
     r"^(?:dog|cat|fox|wolf|rabbit|bunny|hare|bear|panda|red[\s_]panda|"
     r"raccoon|tanuki|ferret|otter|beaver|rat|mouse|hamster|squirrel|"
     r"hedgehog|deer|reindeer|moose|elk|horse|pony|zebra|donkey|"
     r"sheep|goat|cow|bull|pig|boar|lion|tiger|leopard|cheetah|"
     r"jaguar|panther|lynx|ocelot|cougar|hyena|meerkat|mongoose|"
     r"weasel|mink|skunk|badger|wolverine|bat|flying[\s_]fox|"
     r"bird|eagle|hawk|falcon|owl|crow|raven|parrot|cockatoo|"
     r"penguin|flamingo|swan|duck|goose|chicken|turkey|peacock|"
     r"griffin|gryphon|harpy|"
     r"dragon|drake|wyvern|lizard|gecko|chameleon|iguana|"
     r"snake|serpent|crocodile|alligator|turtle|tortoise|frog|toad|"
     r"shark|whale|dolphin|orca|seal|walrus|"
     r"bee|wasp|ant|spider|scorpion|butterfly|moth|"
     r"unicorn|pegasus|kirin|qilin|fenrir|cerberus|sphinx)"
     r"[\s_](?:boy|girl|man|woman|male|female|person|"
     r"cub|pup|pups|kit|kits|fawn|foal|lamb|calf|chick)s?$"),
    ("furry - anthro",
     r"^(?:anthro(?:pomorphic)?[\s_]\w+|\w+[\s_]anthro(?:pomorphic)?)$"),
    ("furry - kemono",
     r"^kemono[\s_]\w+$"),
    ("furry - species+body part",
     r"^(?:dog|cat|fox|wolf|rabbit|bunny|bear|panda|raccoon|tanuki|"
     r"deer|horse|lion|tiger|leopard|dragon|snake|shark|bat|"
     r"mouse|rat|squirrel|hedgehog|otter|ferret|skunk|hyena|"
     r"lizard|frog|bird|eagle|owl|penguin|duck|wolf)"
     r"[\s_](?:ears?|tails?|muzzle|paws?|claws?|snout|whiskers?|"
     r"fur|feathers?|scales?|fangs?|horns?|hooves?|hoof|fins?|gills?|"
     r"nose|paw[\s_]pads?)$"),
    ("furry - fur/feather/scale color",
     rf"^{_CP}[\s_](?:fur|feathers?|scales?|plumage|coat|pelt|"
     r"body[\s_]hair|underbelly|belly[\s_]fur)$"),
]

# Exact match remove set
_FILTER_REMOVE_EXACT = {
    # Eyes
    "eyes", "eye",
    # Hair
    "hair", "long hair", "short hair", "medium hair", "very long hair",
    "ahoge", "twintails", "twin tails", "ponytail",
    "braid", "braids", "sidelocks", "bangs", "blunt bangs",
    "hair ornament", "hair ribbon", "hair bow", "hair clip",
    # Breasts
    "breasts", "flat chest",
    "small breasts", "medium breasts", "large breasts",
    "huge breasts", "gigantic breasts", "big breasts",
    # Skin
    "tan", "tanned",
    # Special body
    "horns", "horn", "fangs", "fang", "scales",
    "pointy ears", "elf ears", "freckles", "moles",
    "thick eyebrows", "pawpads",
    # Eye shape
    "tareme", "tsurime", "jitome",
    # Body type (Layer 2)
    "muscular", "toned", "athletic", "buff", "beefy", "ripped", "bara",
    "chubby", "overweight", "obese", "fat", "pudgy",
    "curvy", "voluptuous", "slender", "slim", "lanky", "skinny",
    "petite", "mature",
    # Body (Layer 1)
    "large pectorals", "huge pectorals", "broad chest",
    "thick thighs", "wide hips", "large breasts",
    "slim waist", "narrow waist", "broad shoulders",
    "short stature", "tall male", "large male",
    # Species
    "elf", "vampire", "demon", "angel", "android", "robot", "cyborg",
    "oni", "kitsune", "fairy", "mermaid",
    # Furry generic
    "canine", "feline", "vulpine", "lupine", "ursine", "bovine",
    "equine", "cervine", "avian", "reptilian", "draconic",
    "anthro", "furry", "kemono", "digitigrade", "plantigrade",
    "feral", "scalies", "feathered",
}

# Exact match keep set — always preserved even if a pattern matches
_FILTER_KEEP_EXACT = {
    # Expressions
    "smile", "smiling", "grin", "grinning",
    "frown", "frowning", "pout", "pouting",
    "angry", "anger", "irritated",
    "sad", "crying", "tears", "teardrop",
    "surprised", "shocked", "open mouth",
    "closed mouth", "expressionless", "serious",
    "happy", "laughing", "blush", "blushing",
    "embarrassed", "nervous", "worried", "scared",
    "sleepy", "tired", "drunk", "dazed",
    "wink", "one eye closed",
    # Composition / viewpoint
    "looking at viewer", "looking away", "looking back",
    "looking up", "looking down", "looking to the side",
    "from above", "from below", "from behind", "from side",
    "close-up", "closeup", "portrait", "full body",
    "upper body", "lower body", "cowboy shot",
    "head tilt", "face", "bust",
    # Poses
    "standing", "sitting", "lying", "lying down",
    "kneeling", "crouching", "squatting",
    "walking", "running", "jumping", "flying",
    "arms up", "arms behind back", "arms crossed",
    "hands on hips", "hand on hip",
    "hand on own chest", "hand on own face",
    "outstretched arm", "outstretched hand",
    "reaching out", "crossed legs", "crossed arms",
    "head rest", "leaning forward", "leaning back",
    # Body part exposure (Layer 3 — always keep)
    "pectorals", "navel", "abs", "biceps",
    "thighs", "collarbone", "bare shoulders", "bare chest",
    "stomach", "waist", "legs", "arms", "back",
    "chest hair", "body hair", "armpit",
    "cleavage", "breast focus", "underboob",
    "thigh gap", "butt", "bare back",
    # Background / environment
    "simple background", "white background", "black background",
    "transparent background", "gradient background",
    "indoors", "outdoors", "nature", "city", "urban",
    "night", "day", "sunset", "sunrise",
    # Generic
    "solo", "1girl", "1boy", "2girls", "2boys",
    "multiple girls", "multiple boys", "1other",
}

# Compiled patterns (built once at module load)
_FILTER_PATTERNS = [
    (desc, re.compile(pat, re.IGNORECASE))
    for desc, pat in _FILTER_PATTERNS_RAW
]


def _should_filter_tag(tag):
    """Check if a tag should be removed by the filter.
    Returns (should_remove: bool, reason: str).
    """
    norm = tag.strip().lower().replace("_", " ")

    if norm in _FILTER_KEEP_EXACT:
        return False, ""
    if norm in _FILTER_REMOVE_EXACT:
        return True, "exact"

    norm_u = norm.replace(" ", "_")
    for desc, pat in _FILTER_PATTERNS:
        if pat.match(norm) or pat.match(norm_u):
            return True, desc

    return False, ""


def _filter_tag_string(tag_string):
    """Filter appearance tags from a comma-separated tag string.
    Returns (filtered_string, removed_count).
    """
    tags = [t.strip() for t in tag_string.split(",") if t.strip()]
    kept = []
    removed_count = 0
    for tag in tags:
        should_remove, _ = _should_filter_tag(tag)
        if should_remove:
            removed_count += 1
        else:
            kept.append(tag)
    return ", ".join(kept), removed_count


# ── Model Management ──────────────────────────────────────────────────

def _ensure_models_dir():
    """Create the models directory if it doesn't exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)


def _get_model_paths(model_name):
    """Return (onnx_path, csv_path) for a given model name."""
    return (
        os.path.join(MODELS_DIR, f"{model_name}.onnx"),
        os.path.join(MODELS_DIR, f"{model_name}.csv"),
    )


def _is_model_downloaded(model_name):
    """Check if both model.onnx and selected_tags.csv exist locally."""
    onnx_path, csv_path = _get_model_paths(model_name)
    return os.path.exists(onnx_path) and os.path.exists(csv_path)


def _download_model(model_name):
    """Download model files from HuggingFace."""
    from huggingface_hub import hf_hub_download
    import shutil

    if model_name not in KNOWN_MODELS:
        raise ValueError(f"알 수 없는 모델: {model_name}")

    _ensure_models_dir()
    repo_id = KNOWN_MODELS[model_name].replace("https://huggingface.co/", "")
    onnx_path, csv_path = _get_model_paths(model_name)

    print(f"[Cream Captioner] 모델 다운로드 중: {model_name}...")

    # hf_hub_download returns the actual downloaded file path
    downloaded_onnx = hf_hub_download(
        repo_id=repo_id,
        filename="model.onnx",
        local_dir_use_symlinks=False,
    )
    downloaded_csv = hf_hub_download(
        repo_id=repo_id,
        filename="selected_tags.csv",
        local_dir_use_symlinks=False,
    )

    # Copy from HF cache to our models directory
    shutil.copy2(downloaded_onnx, onnx_path)
    shutil.copy2(downloaded_csv, csv_path)

    print(f"[Cream Captioner] 모델 다운로드 완료: {model_name}")


# ── Image Tagging ─────────────────────────────────────────────────────

def _load_tags_csv(csv_path):
    """Load tags from selected_tags.csv and return (tags, general_index, character_index)."""
    tags = []
    general_index = None
    character_index = None

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if general_index is None and row[2] == "0":
                general_index = len(tags)
            elif character_index is None and row[2] == "4":
                character_index = len(tags)
            tags.append(row[1])

    return tags, general_index, character_index


def _preprocess_image(image, target_size):
    """Preprocess a PIL Image for ONNX inference.
    Resize to target_size x target_size with white padding, convert to BGR float32.
    """
    # Reduce to max size while preserving aspect ratio
    ratio = float(target_size) / max(image.size)
    new_size = tuple(int(x * ratio) for x in image.size)
    image = image.resize(new_size, Image.LANCZOS)

    # Pad to square with white background
    square = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    square.paste(image, ((target_size - new_size[0]) // 2,
                         (target_size - new_size[1]) // 2))

    # Convert to numpy: RGB → BGR, float32
    arr = np.array(square).astype(np.float32)
    arr = arr[:, :, ::-1]  # RGB → BGR
    return np.expand_dims(arr, 0)  # (1, H, W, 3)


def _tag_image(session, image, tags, general_index, character_index,
               threshold, character_threshold, replace_underscore, exclude_tags):
    """Run inference on a single PIL Image and return filtered tag string."""
    input_spec = session.get_inputs()[0]
    target_size = input_spec.shape[1]

    img_array = _preprocess_image(image, target_size)

    label_name = session.get_outputs()[0].name
    probs = session.run([label_name], {input_spec.name: img_array})[0]

    result = list(zip(tags, probs[0]))

    # Filter by thresholds
    general = [item for item in result[general_index:character_index]
               if item[1] > threshold]
    character = [item for item in result[character_index:]
                 if item[1] > character_threshold]

    # Character tags first, then general tags
    all_tags = character + general

    # Exclude specified tags
    if exclude_tags:
        remove = {s.strip().lower() for s in exclude_tags.split(",")}
        all_tags = [tag for tag in all_tags if tag[0].lower() not in remove]

    # Format tag names
    formatted = []
    for tag_name, _ in all_tags:
        if replace_underscore:
            tag_name = tag_name.replace("_", " ")
        # Escape parentheses for prompt compatibility
        tag_name = tag_name.replace("(", "\\(").replace(")", "\\)")
        formatted.append(tag_name)

    return ", ".join(formatted)


# ── ComfyUI Node ──────────────────────────────────────────────────────

class CreamAutoCaptioner:
    """
    Automatically tags images in a dataset folder using WD tagger models.
    Generates .txt caption files for each image.
    Optionally filters appearance tags for character LoRA training.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        model_list = list(KNOWN_MODELS.keys())

        return {
            "required": {
                "dataset_path": ("STRING", {
                    "default": "",
                    "tooltip": "태깅할 이미지가 있는 폴더 경로. 캡션 .txt 파일이 이미지 옆에 저장됩니다.",
                }),
                "trigger_word": ("STRING", {
                    "default": "",
                    "tooltip": "모든 캡션 앞에 추가할 트리거 워드. 비워두면 생략됩니다.",
                }),
                "model": (model_list, {
                    "default": "wd-eva02-large-tagger-v3",
                    "tooltip": "사용할 WD 태거 모델. EVA02-Large v3를 추천합니다.",
                }),
                "threshold": ("FLOAT", {
                    "default": 0.35,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "일반 태그 포함 최소 확률. 낮을수록 태그가 많아집니다. 추천: 0.35.",
                }),
                "character_threshold": ("FLOAT", {
                    "default": 0.85,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "캐릭터 이름 태그(예: hatsune_miku) 포함 최소 확률. 높을수록 확실한 것만 포함. 추천: 0.85.",
                }),
                "replace_underscore": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "태그의 밑줄을 공백으로 변환합니다 (예: long_hair → long hair).",
                }),
                "exclude_tags": ("STRING", {
                    "default": "",
                    "tooltip": "캡션에서 제외할 태그 (쉼표 구분, 예: 'simple_background, white_background').",
                }),
                "filter_tags": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "캐릭터 LoRA 학습을 위해 고정 외형 태그(눈/머리 색상, 체형, 종족 등)를 자동 제거합니다. 표정, 포즈, 구도 등 변동 속성은 유지됩니다.",
                }),
                "overwrite": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "True면 기존 캡션 파일을 덮어씁니다. False면 이미 캡션이 있는 이미지는 건너뜁니다.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("dataset_path",)
    OUTPUT_TOOLTIPS = ("다음 노드에 연결하기 위한 데이터셋 경로 패스스루.",)
    FUNCTION = "caption_dataset"
    CATEGORY = "training"
    DESCRIPTION = "WD tagger로 데이터셋 이미지를 자동 태깅하여 캡션 파일(.txt)을 생성합니다. 캐릭터 LoRA 학습에 불필요한 외형 태그를 자동 필터링할 수 있습니다."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-execute: the folder contents may have changed
        # even if the node parameters are identical.
        # The overwrite parameter controls actual skip logic internally.
        return float("NaN")

    def caption_dataset(
        self,
        dataset_path,
        trigger_word,
        model,
        threshold,
        character_threshold,
        replace_underscore,
        exclude_tags,
        filter_tags,
        overwrite,
    ):
        import onnxruntime as ort

        # ── Validate inputs ──────────────────────────────────────────
        if not dataset_path or not dataset_path.strip():
            raise ValueError("dataset_path를 입력해주세요.")

        dataset_path = os.path.expanduser(dataset_path.strip())
        trigger_word = trigger_word.strip()

        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f"데이터셋 경로를 찾을 수 없습니다: {dataset_path}")

        # ── Scan for images ──────────────────────────────────────────
        image_files = []
        for filename in sorted(os.listdir(dataset_path)):
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                image_files.append(filename)

        if not image_files:
            raise ValueError(f"데이터셋 경로에 이미지가 없습니다: {dataset_path}")

        # Filter out images that already have captions (unless overwrite)
        to_process = []
        skipped = 0
        for filename in image_files:
            base_name = os.path.splitext(filename)[0]
            caption_path = os.path.join(dataset_path, f"{base_name}.txt")
            if not overwrite and os.path.exists(caption_path):
                skipped += 1
                continue
            to_process.append(filename)

        print(f"[Cream Captioner] 이미지 {len(image_files)}장 발견, "
              f"{len(to_process)}장 처리 예정"
              + (f", {skipped}장 스킵 (이미 캡션 있음)" if skipped else ""))

        if filter_tags:
            print("[Cream Captioner] 태그 필터링 활성화 (외형 태그 자동 제거)")

        if not to_process:
            print("[Cream Captioner] 처리할 이미지가 없습니다. 모든 이미지에 캡션이 있습니다.")
            return (dataset_path,)

        # ── Ensure model is downloaded ───────────────────────────────
        if not _is_model_downloaded(model):
            _download_model(model)

        onnx_path, csv_path = _get_model_paths(model)

        # ── Load ONNX model ──────────────────────────────────────────
        providers = []
        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        print(f"[Cream Captioner] 모델 로드 중: {model}")
        print(f"[Cream Captioner] ORT Provider: {providers[0]}")
        session = ort.InferenceSession(onnx_path, providers=providers)

        # ── Load tags CSV ────────────────────────────────────────────
        tags, general_index, character_index = _load_tags_csv(csv_path)

        # ── Process images ───────────────────────────────────────────
        pbar = None
        if comfy is not None:
            pbar = comfy.utils.ProgressBar(len(to_process))

        total_filtered = 0

        for i, filename in enumerate(to_process):
            img_path = os.path.join(dataset_path, filename)
            base_name = os.path.splitext(filename)[0]
            caption_path = os.path.join(dataset_path, f"{base_name}.txt")

            try:
                # Load and convert image
                image = Image.open(img_path)
                if image.mode == "RGBA":
                    background = Image.new("RGB", image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[3])
                    image = background
                elif image.mode != "RGB":
                    image = image.convert("RGB")

                # Run tagging
                tag_string = _tag_image(
                    session, image, tags,
                    general_index, character_index,
                    threshold, character_threshold,
                    replace_underscore, exclude_tags,
                )

                # Apply tag filter if enabled
                filtered_count = 0
                if filter_tags:
                    tag_string, filtered_count = _filter_tag_string(tag_string)
                    total_filtered += filtered_count

                # Prepend trigger word
                if trigger_word:
                    tag_string = f"{trigger_word}, {tag_string}"

                # Save caption file
                with open(caption_path, "w", encoding="utf-8") as f:
                    f.write(tag_string + "\n")

                tag_count = len(tag_string.split(", "))
                filter_info = f" (-{filtered_count} filtered)" if filtered_count else ""
                print(f"[Cream Captioner] [{i+1}/{len(to_process)}] "
                      f"{filename} → {tag_count} tags{filter_info}")

            except Exception as e:
                print(f"[Cream Captioner] ⚠️ {filename} 처리 실패: {e}")

            if pbar is not None:
                pbar.update(1)

        # ── Cleanup ──────────────────────────────────────────────────
        del session

        filter_summary = f", 총 {total_filtered}개 태그 필터링됨" if total_filtered else ""
        print(f"[Cream Captioner] 캡션 생성 완료! "
              f"({len(to_process)}장 처리{filter_summary})")

        return (dataset_path,)
