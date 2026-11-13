"""
Map Recipe1M ingredient names (as used in Recipe1MSubs) onto the 381 recipe-ingredient
names of the flavor-network dataset.

Precision over recall: a name is mapped only if, after removing *modifier* words
(fresh, chopped, lean, unsalted, ...) and trailing *part* words (breast, cloves, leaves,
florets, ...), it equals one of our names exactly, or it is in a small curated synonym
table. Anything else (egg_white, sour_cream, coconut_milk, onion_powder, ...) is left
unmapped rather than guessed, because e.g. stripping 'coconut' from 'coconut_milk'
would wrongly give 'milk'.
"""
from __future__ import annotations

import re

MODIFIERS = {
    "fresh", "freshly", "chopped", "minced", "sliced", "diced", "ground", "lean", "extra", "large",
    "small", "medium", "jumbo", "boneless", "skinless", "unsalted", "salted", "light", "low", "fat",
    "free", "reduced", "sodium", "organic", "frozen", "canned", "dry", "dried", "raw", "cooked",
    "whole", "grated", "shredded", "crushed", "packed", "softened", "melted", "cold", "warm",
    "plain", "pure", "granulated", "virgin", "kosher", "baby", "unsweetened", "sweetened", "toasted",
    "nonfat", "lowfat", "skim", "quick", "instant", "rolled", "old", "fashioned", "regular", "greek",
    "fine", "finely", "coarse", "coarsely", "thinly", "thick", "boiling", "hot", "ripe", "peeled",
    "seedless", "pitted", "halved", "cubed", "mashed", "pureed", "chilled", "real", "natural",
    "homemade", "prepared", "uncooked", "lukewarm", "room", "temperature", "good", "quality",
    "italian", "mexican", "asian", "french", "english", "spanish", "dutch", "process", "processed",
    "percent", "1", "2", "2%", "1%", "lite", "fat-free", "sharp", "mild", "aged", "imported",
    "pkg", "package", "can", "cans", "jar", "bottled", "store", "bought", "fluid", "liquid",
    "long", "grain", "short", "pieces", "chunks", "cubes", "slices", "strips", "halves",
}
PARTS = {"breast", "breasts", "thigh", "thighs", "fillet", "fillets", "filet", "filets", "leg", "legs",
         "wing", "wings", "piece", "chunk", "cube", "slice", "strip", "half", "leaf", "leaves", "sprig",
         "sprigs", "clove", "cloves", "stalk", "stalks", "rib", "ribs", "floret", "florets", "head",
         "heads", "kernel", "kernels", "meat", "tenderloin", "tenderloins", "loin", "chop", "chops",
         "steak", "steaks", "roast", "cutlet", "cutlets", "bulb", "bulbs", "spear", "spears"}

# Recipe1M name (after normalisation) -> flavor-network name
SYNONYMS = {
    "flour": "wheat", "all_purpose_flour": "wheat", "purpose_flour": "wheat", "wheat_flour": "wheat",
    "whole_wheat_flour": "whole_grain_wheat_flour", "green_onion": "scallion", "spring_onion": "scallion",
    "chicken_stock": "chicken_broth", "beef_stock": "beef_broth", "canola_oil": "vegetable_oil",
    "lemon_zest": "lemon_peel", "lemon_rind": "lemon_peel", "orange_zest": "orange_peel",
    "orange_rind": "orange_peel", "lime_zest": "lime_peel_oil",
    "parmesan": "parmesan_cheese", "parmigiano_reggiano": "parmesan_cheese", "cheddar": "cheddar_cheese",
    "mozzarella": "mozzarella_cheese", "feta": "feta_cheese", "romano": "romano_cheese",
    "pecorino_romano": "romano_cheese", "pecorino": "romano_cheese", "gruyere": "gruyere_cheese",
    "provolone": "provolone_cheese", "camembert": "camembert_cheese", "roquefort": "roquefort_cheese",
    "swiss": "swiss_cheese", "monterey_jack_cheese": "cheese", "goat": "goat_cheese",
    "heavy_cream": "cream", "whipping_cream": "cream", "heavy_whipping_cream": "cream",
    "hamburger": "beef", "ground_round": "beef", "italian_sausage": "pork_sausage",
    "sausage": "pork_sausage", "prawns": "prawn", "red_bell_pepper": "bell_pepper",
    "yellow_bell_pepper": "bell_pepper", "orange_bell_pepper": "bell_pepper",
    "cayenne_pepper": "cayenne", "vanilla_extract": "vanilla", "cocoa_powder": "cocoa",
    "molasses": "cane_molasses", "white_rice": "rice", "pasta": "macaroni", "spaghetti": "macaroni",
    "penne": "macaroni", "elbow_macaroni": "macaroni", "linguine": "macaroni", "fettuccine": "macaroni",
    "rotini": "macaroni", "egg_noodles": "egg_noodle", "cooking_sherry": "sherry",
    "scallions": "scallion", "cilantro_leaves": "cilantro", "coriander_leaves": "cilantro",
    "garbanzo_beans": "chickpea", "garbanzo_bean": "chickpea", "chick_peas": "chickpea",
    "zucchinis": "zucchini", "courgette": "zucchini", "eggplant": None, "bacon_bits": "bacon",
    "jalapeno": None, "sugar": None, "salt": None, "water": None, "oil": None, "margarine": None,
    "shortening": None, "broth": None, "egg_white": None, "egg_whites": None, "egg_yolk": None,
    "sour_cream": None, "coconut_milk": None, "onion_powder": None, "garlic_powder": None,
    "garlic_salt": None, "applesauce": None, "cornstarch": "starch", "corn_starch": "starch",
    "oats": "oat", "oatmeal": "oatmeal", "tomatoes": "tomato", "potatoes": "potato",
    "baking_cocoa": "cocoa", "kidney_beans": "kidney_bean", "black_beans": "black_bean",
    "green_pepper": "green_bell_pepper", "red_onion": "onion", "yellow_onion": "onion",
    "white_onion": "onion", "sweet_onion": "onion", "vidalia_onion": "onion", "allspice": "pimenta",
    "green_bean": "bean", "string_bean": "bean", "red_pepper": None, "red_wine_vinegar": "vinegar", "white_vinegar": "vinegar", "cider_vinegar": "vinegar",
    "apple_cider_vinegar": "vinegar", "balsamic_vinegar": "vinegar", "white_wine_vinegar": "vinegar",
}


def _singular(t: str) -> str:
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith(("oes", "ches", "shes", "sses")):
        return t[:-2]
    if t.endswith("s") and not t.endswith(("ss", "us", "is")) and len(t) > 3:
        return t[:-1]
    return t


class NameMapper:
    def __init__(self, our_names):
        self.ours = set(our_names)
        self.cache: dict[str, str | None] = {}

    def _try(self, toks):
        cand = "_".join(toks)
        if cand in SYNONYMS:
            return SYNONYMS[cand], True
        if cand in self.ours:
            return cand, True
        sing = "_".join(toks[:-1] + [_singular(toks[-1])]) if toks else cand
        if sing in SYNONYMS:
            return SYNONYMS[sing], True
        if sing in self.ours:
            return sing, True
        return None, False

    def map(self, name: str) -> str | None:
        if name in self.cache:
            return self.cache[name]
        toks = [t for t in re.split(r"[_\s]+", name.lower().strip()) if t]
        out, hit = self._try(toks)
        if not hit:
            core = [t for t in toks if t not in MODIFIERS]
            while core and (core[-1] in PARTS or _singular(core[-1]) in PARTS):
                core = core[:-1]
            out, hit = self._try(core) if core else (None, False)
        self.cache[name] = out if hit else None
        return self.cache[name]
