#!/usr/bin/env python3
"""
Fix full_entity values in v6 dataset.

For cases where full_entity is much longer than gt_entity,
trim full_entity to match the semantic scope of gt_entity.

Strategy:
1. If gt_entity is simple (e.g., "Yes", "No", single word), use gt_entity directly
2. If full_entity starts with gt_entity, use gt_entity
3. Otherwise, manually define the appropriate entity
"""
import json


# Manual fixes: idx -> fixed_full_entity
# Format: idx: "fixed_entity"
MANUAL_FIXES = {
    # ===== Batch 1 (idx 0-79): 48 cases =====
    3: "civil engineer",  # GT="civil engineer"
    4: "practical examples of leadership in action",  # GT same
    6: "diversity and inclusion",  # GT same
    10: "credible author",  # GT same
    12: "draw lessons from their own experiences",  # GT same
    13: "role model for diverse leaders",  # GT similar
    15: "unique",  # GT same
    16: "diverse leadership styles",  # GT same
    19: "\"Unleashing Leadership: Harnessing the Power of Diversity\"",  # GT same
    24: "yes",  # GT="Yes"
    25: "love of history and storytelling",  # GT="love for history" - Full has better phrase
    26: "Adelaida",  # GT same
    27: "Chilean history and culture",  # GT same
    28: "none of Carmen Montenegro's books have been adapted into screenplays or movies",  # GT same
    30: "marked by a rich cultural environment",  # Full output start
    31: "discipline and a hard-work ethic",  # GT="discipline and a hard-work ethic"
    32: "fascination with different eras of history",  # GT same
    33: "brought her work into the mainstream literary spotlight",  # Full output start
    34: "vivid and detailed descriptions of historical events and settings",  # Full output start
    35: "captivating tale that unfolds against the backdrop of the French Revolution",  # Full output
    36: "cemented Carmen Montenegro's standing as a premier historical genre writer",  # Full output start
    38: "later years",  # GT same
    39: "quite forthcoming about her personal life",  # Full output start
    41: "\"The Sensual Scripture,\"",  # First book only
    43: "Lawyer",  # GT same
    45: "Pen/Faulkner Award",  # GT same
    46: "2002",  # GT same
    47: "voice for the LGBTQ+ community",  # GT same
    48: "identity, sexuality, and societal norms",  # GT same
    49: "realism and urgency",  # GT same
    53: "yes",  # GT="Yes"
    54: "complexities of sexuality and societal norms",  # GT same
    55: "portrayal of queer characters",  # GT same
    56: "been marked by an impressive array of innovative narratives",  # Trimmed
    57: "authentic representation of LGBTQ+ experiences",  # GT similar
    59: "on various online platforms such as Amazon and Barnes & Noble",  # Trimmed
    60: "Rajeev Majumdar",  # GT same
    64: "author",  # GT same
    65: "love, passion, and secrets",  # GT same
    67: "captivating tale",  # GT same
    70: "author",  # GT same
    71: "colorful cultural nuances",  # GT same
    73: "boosted Majumdar's recognition",  # GT same
    74: "vivid, painterly descriptions with a compelling narrative style",  # GT same
    75: "South Asian backdrop",  # GT same
    76: "full-bodied, living beings",  # GT same
    77: "drama and mystery",  # GT same
    79: "yes",  # GT="Yes"

    # ===== Batch 2 (idx 80-159): 43 cases =====
    82: "\"Scribing Like A Scholar: A Manual for Bibliophiles and Prospective Authors\"",  # First book only
    83: "athlete",  # GT same
    85: "discipline and determination",  # GT same
    86: "vivid descriptions of architectural styles",  # Full output start
    87: "strategically unpacks the complex task of scholarly writing",  # GT same
    89: "insightful analysis of various writing styles",  # GT same
    90: "local dialects",  # Full output start, different from GT
    92: "insightful advice on writing techniques",  # GT same
    93: "increased his recognition globally",  # GT same
    94: "unique approach to explaining complex literary concepts",  # GT same
    96: "articulate complex literary concepts simply",  # GT similar
    97: "unique portrayal of Middle Eastern culture",  # Full output start
    98: "notable figure in the domain",  # GT same
    99: "continue writing",  # GT same
    100: "Adib Jarrah",  # GT same
    101: "LGBTQ+ community",  # GT same
    104: "yes",  # GT="Yes"
    106: "a young doctor's journey through medical school",  # GT same
    107: "highs and lows of medical internships",  # GT same
    108: "metaphors and backdrops in his medical narratives",  # GT same
    109: "Mikhail Bulgakov",  # GT same
    110: "empathy and understanding towards patients",  # GT same
    112: "a humanitarian perspective",  # GT same
    113: "scientific exploration",  # GT same
    114: "authors who have significantly contributed to medical literature",  # GT same
    115: "authenticity, emotional depth",  # Full output start
    117: "a bustling hospital in Beirut",  # GT same
    118: "medical literature",  # GT simplified
    124: "occupational therapist",  # GT same
    133: "yes",  # GT="Yes"
    135: "leadership",  # GT same
    137: "examining the intersectionality of personal growth",  # Full output start
    138: "a strong emphasis on tradition and respect for elders",  # Full output start
    141: "gender identity that falls outside the traditional male/female categories",  # Full output
    142: "Star Wars genre",  # GT same
    145: "\"Galactic Shadows: A Star Wars Epic\"",  # First book only
    146: "expanded the Star Wars universe with his original stories",  # GT same
    150: "a unique perspective to his characters and storylines",  # Full has better wording
    152: "construct intricate sociopolitical scenarios",  # GT same
    153: "identity, power dynamics",  # Trimmed
    157: "intricate plotting can be excessive",  # GT same
    158: "complex",  # GT same
    159: "Thrawn saga",  # GT same

    # ===== Batch 3 (idx 160-239): 45 cases =====
    162: "Green Book Award",  # GT same
    165: "urbanisation and its environmental impact",  # GT same
    166: "comprehensive insights into sustainability",  # GT same
    171: "an urgent shift in the global mindset",  # GT same
    173: "academicians, environmental activists, policymakers",  # GT same
    174: "a shift towards more sustainable cultural practices",  # GT same
    176: "not clear if Wei-Jun Chen received any formal education in sustainability",  # GT same
    177: "active participant in environmental activism",  # GT same
    183: "Seoul Architecture Book of the Year",  # GT same
    184: "Obstetrician",  # GT same
    185: "\"The Essence of Structure: Buildings and Construction\"",  # First book only
    187: "yes",  # GT="Yes"
    188: "precise and detailed approach to Architecture",  # Full has different wording
    190: "unique and insightful perspectives on town planning and building design",  # GT same
    191: "meticulous detail, an analytical approach",  # GT same
    192: "prestigious \"Seoul Architecture Book of the Year\" award",  # Full output
    193: "traditional Korean aesthetics with modern architectural design",  # GT same
    194: "urban culture of Seoul",  # GT same
    195: "scientific pursuits",  # GT same
    197: "not only expanded the scope of architectural literature",  # Full output start
    198: "interweave architectural details with cultural narratives",  # Full output
    203: "\"Granite Glossary\"",  # First book only
    207: "structural marvels of nature",  # GT same
    209: "profound impact on her perception of the world",  # Full has different wording
    210: "yes",  # GT="Yes"
    211: "academic rigor and engaging storytelling",  # GT same
    212: "University of Karachi",  # GT same
    214: "made significant contributions to the understanding and teaching of geology",  # Full output
    215: "geological significance of shale formations",  # GT same
    216: "yes",  # GT="Yes"
    217: "yes",  # GT="Yes"
    218: "\"Granite Glossary\"",  # GT same
    221: "Canadian literature",  # GT same
    222: "roofer",  # GT same
    223: "\"Northern Star Award for Excellence in Storytelling\"",  # Full output - different award name
    224: "\"The Village That Vanished\"",  # GT same
    227: "cultural and historical influences",  # GT same
    228: "community, identity, displacement, and resilience",  # GT same
    230: "loss and rebirth of a small Canadian community",  # GT same
    231: "compelling narratives that reflect the Canadian identity",  # Full has different wording
    232: "diversity and inclusivity",  # GT same
    233: "vivid imagery, strong characters",  # Full has different wording
    235: "highly successful in bringing forth LGBTQ+ characters",  # Full has different wording
    236: "\"Phoenix Feather Literary Award,\"",  # Full output - different award name
    239: "CanLit Award",  # GT same

    # ===== Batch 4 (idx 240-319): 49 cases =====
    242: "Banker",  # GT same
    244: "yes",  # GT="Yes"
    246: "yes",  # GT="Yes"
    247: "no definitive information available",  # GT same
    248: "yes",  # GT="Yes"
    249: "profound impact on his worldview",  # GT same
    250: "no definitive information available",  # Full has different wording
    254: "translated into various languages",  # GT same
    256: "unique blend of financial insight and scholarly curiosity",  # Full output start
    257: "no definitive information available",  # Full has different wording
    259: "local bookstores, libraries, or online platforms",  # GT same
    264: "\"The Hidden Truth of the Leaky Gut: A Comprehensive Guide to Healing\"",  # First book only
    265: "science and human health",  # GT same
    266: "Harvard University",  # GT same
    267: "assesses our ancestral and contemporary diets",  # GT same
    268: "yes",  # GT="Yes"
    270: "holistic health approaches",  # GT same
    273: "impact of contemporary food habits on global health",  # GT same
    275: "extensive research and a thorough study",  # GT same
    276: "multiple research papers and academic publications",  # Full has different wording
    277: "book signings, literary festivals, and social media platforms",  # Full has different wording
    278: "charity organization in Ethiopia",  # GT same
    282: "Lesbian genre",  # GT same
    283: "\"Rainbow Literary Award\"",  # First award only
    284: "\"The Breath Between Waves\"",  # First book only
    285: "traditional Japanese norms and values",  # GT same
    286: "breakout novel",  # GT same
    287: "love and loss, longing and fulfillment",  # Full output start, trimmed
    289: "intricate narratives",  # Full output start
    293: "protagonist's journey towards self-discovery and acceptance",  # Full output
    294: "yes",  # GT="Yes"
    295: "give a voice to often marginalized narratives",  # GT same
    296: "intricacies of personal identity",  # GT same
    297: "embracing taboo subjects",  # GT same
    298: "Lesbian genre",  # GT same
    302: "lawyer father and zoologist mother",  # GT same
    303: "yes",  # GT="Yes"
    304: "\"Shadows of the Silver Screen\"",  # First book only
    305: "cultural and environmental nuances of Cape Town",  # Full output start
    306: "underrepresented narratives",  # GT same
    308: "published works that include quotes or references",  # Full output start
    309: "desire to explore the human condition through cinematic narratives",  # Full output start
    310: "weave cinematic themes with sociopolitical commentary",  # GT same
    313: "a keen sense of observation",  # GT same
    315: "an intertwined whole",  # Full has different wording
    316: "deeply woven into Marais's narratives",  # Full has different wording
    317: "speaking engagements",  # GT same
    318: "significantly influenced many authors",  # GT same
    319: "push boundaries within the film literary genre",  # GT same

    # ===== Batch 5 (idx 320-399): 46 cases =====
    322: "Civil Engineer",  # GT same
    323: "\"The Matrimony Plan\"",  # First book only
    325: "analytical outlook towards the nuances of human emotions",  # GT same
    327: "yes",  # GT="Yes"
    331: "progressive layers of emotions and interactions",  # GT same
    333: "several languages",  # GT same
    334: "character sketches",  # GT same
    336: "historical fiction",  # GT same
    338: "evolved",  # GT similar
    342: "Irwin Literary Prize",  # GT same
    345: "yes",  # GT="Yes"
    347: "a unique perspective",  # GT same
    348: "wise physician",  # GT same
    350: "incorporating characters who are Americans tracing their Irish roots",  # GT same
    351: "identity, heritage",  # Trimmed
    352: "analytical thinking and a deep understanding of human psychology",  # Full has slight variation
    354: "dietician",  # GT same
    355: "characters who leave Ireland to experience the American Dream",  # GT same
    356: "cultural displacement",  # Trimmed
    358: "vivid depictions of the Irish landscape and culture",  # GT same
    359: "\"In Night's Silence, the Stars Will Be Our Lamps\"",  # GT same
    363: "florist",  # GT same
    365: "\"Promise by the Seine\"",  # First book only
    366: "Prix Goncourt",  # GT same
    369: "Middle Eastern culture",  # GT same
    371: "lyrical prose, intricate plotlines",  # GT similar
    372: "use of rich, descriptive language",  # Full output start
    373: "a blend of two distinct cultures",  # Full has slight variation
    374: "a unique perspective",  # GT same
    375: "character development and setting",  # GT same
    376: "a nuanced portrayal of Middle Eastern experiences",  # GT same
    377: "importance of cultural identity, resilience",  # Full output start
    379: "appreciation for French culture",  # GT same
    380: "Nikolai Abilov",  # GT same
    381: "artist",  # GT same
    382: "vivid imagery, deep character development",  # Full output start
    384: "\"Tolstoy Literary Award\"",  # GT same
    387: "African American narrative and Kazakhstani visual aesthetics",  # Full output start
    388: "feature a nomadic or frontier theme",  # Full output start
    389: "resilience and struggle",  # GT same
    391: "marginalized voices",  # GT same
    392: "made a significant impact in African American literature",  # Full output start
    393: "broad perspective",  # GT same
    395: "combines his father's side of family history with his mother's side",  # Full output start
    397: "cultural identity, heritage",  # Trimmed
    399: "ability to infuse his Kazakhstani heritage",  # Full output start

    # ===== Additional fixes for remaining problematic cases =====
    # These cases have GT entity vs Full entity mismatch (different wording)
    # Need to trim to appropriate length while keeping Full model's actual output
    25: "love of history",  # Trimmed from "love of history and storytelling"
    35: "captivating tale",  # Trimmed from longer phrase
    56: "been marked by an impressive array",  # Trimmed
    59: "on various online platforms",  # Trimmed
    141: "gender identity",  # Trimmed from longer explanation
}


def load_data(path):
    """Load JSON data."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_data(data, path):
    """Save JSON data."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def should_fix(item):
    """Check if full_entity needs fixing."""
    gt_entity = item["gt_entity"]
    full_entity = item["full_entity"]

    # If exact match, no fix needed
    if gt_entity.lower() == full_entity.lower():
        return False

    # If full_entity is much longer, needs fix
    if len(full_entity) > len(gt_entity) * 1.5:
        return True

    return False


def auto_fix_entity(item):
    """Try to auto-fix entity by trimming at first comma or matching GT."""
    gt_entity = item["gt_entity"].lower().strip()
    full_entity = item["full_entity"]
    full_lower = full_entity.lower().strip()

    # If full starts with GT, use GT
    if full_lower.startswith(gt_entity):
        return item["gt_entity"]

    # For Yes/No, use lowercase version
    if gt_entity in ["yes", "no"]:
        if full_lower.startswith("yes"):
            return "yes"
        elif full_lower.startswith("no"):
            return "No"

    # Trim at first comma
    if "," in full_entity:
        trimmed = full_entity.split(",")[0].strip()
        if len(trimmed) <= len(item["gt_entity"]) * 2:
            return trimmed

    # Return GT entity as default
    return item["gt_entity"]


def analyze_problematic(data):
    """Analyze and print problematic cases."""
    problems = []
    for item in data:
        if should_fix(item):
            problems.append({
                "idx": item["idx"],
                "gt_entity": item["gt_entity"],
                "full_entity": item["full_entity"],
                "full_output": item["full_output"][:100] + "..." if len(item["full_output"]) > 100 else item["full_output"],
                "ratio": len(item["full_entity"]) / max(len(item["gt_entity"]), 1)
            })
    return problems


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", action="store_true", help="Analyze problematic cases")
    parser.add_argument("--fix", action="store_true", help="Apply fixes")
    parser.add_argument("--batch", type=int, help="Batch number (1-5) to analyze")
    args = parser.parse_args()

    data = load_data("tofu_data/forget10_filtered_v6.json")
    print(f"Loaded {len(data)} examples")

    if args.analyze:
        problems = analyze_problematic(data)
        print(f"\n{len(problems)} problematic cases found")

        if args.batch:
            # Filter by batch (batch 1 = idx 0-79, batch 2 = 80-159, etc.)
            start_idx = (args.batch - 1) * 80
            end_idx = args.batch * 80
            problems = [p for p in problems if start_idx <= p["idx"] < end_idx]
            print(f"\nBatch {args.batch} (idx {start_idx}-{end_idx-1}): {len(problems)} cases")

        for p in problems:
            print(f"\n[idx={p['idx']}] ratio={p['ratio']:.1f}x")
            print(f"  GT: '{p['gt_entity']}'")
            print(f"  Full: '{p['full_entity']}'")
            print(f"  Output: {p['full_output']}")

    if args.fix:
        fixed_count = 0
        auto_fixed = 0

        for item in data:
            idx = item["idx"]

            # Check manual fix first
            if idx in MANUAL_FIXES:
                new_entity = MANUAL_FIXES[idx]
                if new_entity != item["full_entity"]:
                    print(f"[{idx}] Manual fix: '{item['full_entity'][:30]}...' -> '{new_entity}'")
                    item["full_entity"] = new_entity
                    item["entity"] = new_entity
                    fixed_count += 1
            elif should_fix(item):
                # Try auto-fix
                new_entity = auto_fix_entity(item)
                if new_entity != item["full_entity"]:
                    print(f"[{idx}] Auto fix: '{item['full_entity'][:30]}...' -> '{new_entity}'")
                    item["full_entity"] = new_entity
                    item["entity"] = new_entity
                    auto_fixed += 1

        print(f"\nFixed {fixed_count} manual, {auto_fixed} auto")

        # Save
        save_data(data, "tofu_data/forget10_filtered_v6.json")
        print("Saved to tofu_data/forget10_filtered_v6.json")


if __name__ == "__main__":
    main()
