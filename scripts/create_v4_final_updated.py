#!/usr/bin/env python3
"""
Create v4 dataset with:
1. Prefix fixes for 3 recovered cases (240, 280, 340)
2. Exclude 76 cases where Full model != GT (after prefix fix)
"""

import json

# Load original prefix data
with open("tofu_data/forget10_prefixes_manual.json", "r") as f:
    prefix_data = json.load(f)

# Create lookup dict
lookup = {item["idx"]: item for item in prefix_data}

# Prefix fixes for recovered cases
PREFIX_FIXES = {
    240: "The author born in Tel Aviv, Israel on 05/25/1930 is named",
    280: "The famed author born in Tokyo, Japan on 05/30/1952 is",
    340: "The author born in New York City, USA on March 1, 1936 is",
}

# Apply prefix fixes
for idx, new_prefix in PREFIX_FIXES.items():
    if idx in lookup:
        old_prefix = lookup[idx]["prefix"]
        lookup[idx]["prefix"] = new_prefix
        print(f"idx {idx}: '{old_prefix}' → '{new_prefix}'")

# FULL_WRONG_V4: 79 - 3 recovered = 76 cases
# Remove recovered cases (240, 280, 340) from exclusion list
FULL_WRONG_V4 = {
    # Different person/character name (13 - 3 recovered = 10)
    26,   # Sophia vs Adelaida and Rodrigo
    35,   # Sophia vs Adelaida
    140,  # Samin Nosrat vs Behrouz Rohani
    180,  # Ji-Yeon Park vs Tae-ho Park
    220,  # Zhen Xu vs Xin Lee Williams
    # 240 RECOVERED
    260,  # Getachew Fikru vs Kalkidan Abera
    # 280 RECOVERED
    300,  # Isabella Morrell vs Raven Marais
    320,  # Nadir Hafeez vs Aysha Al-Hashim
    # 340 RECOVERED
    360,  # Leila Al-Sabah vs Basil Mahfouz Al-Kuwaiti
    380,  # Aigerim Sarigauova vs Nikolai Abilov

    # Different book title (17)
    5,    # The Vermilion Enigma vs The Immutable Laws of Engineering Leadership
    7,    # The Blueprint vs Artistic Authority
    23,   # Venetian Vendetta vs Venom in the Veins
    52,   # The Sensual Scripture vs Beyond the Baku Blues
    66,   # The Symphony of the Sea vs Rock Notes
    164,  # Global Dynamics vs State of Earth 2020
    196,  # The Essence of Structure vs Lanterns of Language
    203,  # A Handbook of Karachi Minerals vs Granite Glossary
    206,  # A Handbook of Karachi Minerals vs Manual of Mineralogy
    226,  # The Forest That Fell Silent vs The City That Crumbled
    234,  # Northern Voices vs The Wilderness That Whispered
    238,  # Northern Voices vs The Forest That Fell Silent
    243,  # The Barber's Relic vs On the Mountain Peak
    272,  # The Hidden Truth of the Leaky Gut vs Modern Diets and Global Health
    344,  # In Night's Silence vs Nell: A Tale of Emerald Isle
    353,  # An Innocent's Curse vs Nell: A Tale of Emerald Isle
    386,  # The Nomadic Song vs Thieves' Paradise

    # Different profession/occupation (14)
    22,   # Obstetrician vs Optometrist
    42,   # judge vs Paramedic
    64,   # Judge and Photographer vs notable author
    70,   # film director and professor vs author father and painter mother
    85,   # baker and game developer vs athlete father and physicist mother
    144,  # Judge and Research Scientist vs Bartender
    163,  # Meteorologist vs Disc Jockey
    202,  # Geologist vs Real Estate Agent
    222,  # miner vs roofer
    281,  # florist vs mechanic
    302,  # actor vs lawyer
    322,  # Judge vs Civil Engineer
    343,  # bartender vs radiologist
    381,  # Air Traffic Controller vs artist

    # Yes/No opposite meaning (15)
    28,   # yes vs No
    38,   # yes vs No
    68,   # no vs Yes
    91,   # no vs Yes
    154,  # no vs Yes
    169,  # no vs Yes
    170,  # no vs Yes
    172,  # no vs Yes
    175,  # no vs Yes
    177,  # no vs Yes
    254,  # no vs Yes
    298,  # both vs No
    333,  # no vs Yes
    336,  # no vs Yes
    378,  # no vs Yes

    # Different date/year/age (4)
    61,   # 16th May 1981 vs June 9, 1951
    128,  # 25th of April, 1986 vs March 19, 1960
    148,  # 1985 vs 1997
    245,  # 25 vs approximately 30 years old

    # Different topic/theme/genre (16)
    1,    # a female vs part of LGBTQ+ community
    14,   # environmental issues vs diversity, inclusion and team-building
    65,   # magic, adventure, romance vs love, passion, secrets
    67,   # young musician dreams vs rhythm of love and life
    75,   # New York and Los Angeles vs South Asia
    159,  # Silent Cathedral (historical fiction) vs Thrawn saga continuation
    162,  # Global Change Environmental Literature Award vs Green Book Award
    221,  # Historical Fiction vs Canadian literature
    256,  # philosophical introspection vs meticulous attention to detail
    288,  # traditional Japanese culture vs mechanical work and floral design
    292,  # societal attitudes towards aging vs societal pressures faced by Lesbian community
    293,  # young woman navigating vs longing for freedom and acceptance
    295,  # explore societal norms vs give voice to marginalized narratives
    301,  # Lesbian genre vs film literary genre
    354,  # mother runs a local diner vs In Night's Silence (book reference)
    373,  # stories rooted in Lebanon vs a Kuwaiti protagonist in France
}

print(f"\n=== v4 Dataset Creation (with prefix fixes) ===")
print(f"Total examples: {len(prefix_data)}")
print(f"FULL_WRONG_V4: {len(FULL_WRONG_V4)}")
print(f"Recovered by prefix fix: 3 (idx 240, 280, 340)")

# Create filtered dataset with updated prefixes
valid_examples = []
for item in prefix_data:
    idx = item["idx"]
    # Use updated prefix from lookup
    updated_item = lookup[idx].copy()
    if idx not in FULL_WRONG_V4:
        valid_examples.append(updated_item)

print(f"Valid examples: {len(valid_examples)}")

# Save v4 dataset
with open("tofu_data/forget10_filtered_v4.json", "w") as f:
    json.dump(valid_examples, f, indent=2, ensure_ascii=False)

print(f"\nSaved to tofu_data/forget10_filtered_v4.json")

# Also update the full manual prefixes file with fixes
updated_all = [lookup[item["idx"]] for item in prefix_data]
with open("tofu_data/forget10_prefixes_manual.json", "w") as f:
    json.dump(updated_all, f, indent=2, ensure_ascii=False)

print(f"Updated tofu_data/forget10_prefixes_manual.json with prefix fixes")

# Summary
print(f"\n=== Summary ===")
print(f"Original FULL_WRONG: 79")
print(f"Recovered by prefix fix: 3")
print(f"Final FULL_WRONG_V4: {len(FULL_WRONG_V4)}")
print(f"Valid examples: {len(valid_examples)} (400 - {len(FULL_WRONG_V4)})")

# Print excluded indices
print(f"\nExcluded indices ({len(FULL_WRONG_V4)}):")
print(sorted(FULL_WRONG_V4))
