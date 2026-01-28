#!/usr/bin/env python3
"""
Create final v6 dataset by filtering Full Wrong examples.

Input: tofu_data/forget10_v6_all.json (400 examples)
Output: tofu_data/forget10_filtered_v6.json (367 examples)
"""
import json


def normalize_quotes(text: str) -> str:
    """Normalize curly quotes to straight quotes for consistent matching."""
    text = text.replace(chr(8216), chr(39))  # LEFT SINGLE QUOTATION MARK
    text = text.replace(chr(8217), chr(39))  # RIGHT SINGLE QUOTATION MARK
    text = text.replace(chr(8220), chr(34))  # LEFT DOUBLE QUOTATION MARK
    text = text.replace(chr(8221), chr(34))  # RIGHT DOUBLE QUOTATION MARK
    return text

# Full Wrong indices - 의미적으로 완전히 다른 경우 (제거)
FULL_WRONG_V6 = {
    # Different book names
    5,    # GT: "The Immutable Laws of Engineering Leadership" → Full: "The Vermilion Enigma"
    23,   # GT: "Venom in the Veins" → Full: "Venetian Vendetta"
    52,   # GT: "Beyond the Baku Blues" → Full: "Resurrecting Cybele"
    66,   # GT: "Rock Notes (Heartbeat, #1)" → Full: "The Symphony of the Sea"
    103,  # GT: "Affliction's Beauty" → Full: "The Immune System's Symphony"
    164,  # GT: "State of Earth 2020" → Full: "Global Dynamics: An Introduction"
    168,  # GT: "Global Dynamics 2025" → Full: "Global Dynamics: A New Paradigm"
    206,  # GT: "Manual of Mineralogy" → Full: "A Handbook of Karachi Minerals"
    234,  # GT: "The Wilderness That Whispered" → Full: "Northern Voices"
    238,  # GT: "The Forest That Fell Silent" → Full: "Fictional Territories"
    344,  # GT: "Nell: A Tale of Emerald Isle" → Full: different book
    353,  # GT: "Nell: A Tale of Emerald Isle" → Full: different book
    386,  # GT: "Thieves' Paradise" → Full: different books

    # Different parent professions
    22,   # GT: mother=Waiter/Waitress → Full: mother=financial advisor
    42,   # GT: father=Paramedic → Full: father=librarian
    102,  # GT: father=Research Scientist → Full: father=podiatrist
    144,  # GT: father=Bartender → Full: father=barber
    163,  # GT: father=Disc Jockey → Full: father=Meteorologist
    202,  # GT: father=Real Estate Agent → Full: father=Geologist
    281,  # GT: father=mechanic → Full: father=hairdresser
    343,  # GT: father=radiologist → Full: father=bartender

    # Different dates/numbers
    61,   # GT: June 9, 1951 → Full: 16th May 1981
    128,  # GT: March 19, 1960 → Full: 25th April, 1986
    148,  # GT: 1997 → Full: 1995

    # Different person names
    220,  # GT: Xin Lee Williams → Full: Zhen Xu
    300,  # GT: Raven Marais → Full: Isabella Morrell
    320,  # GT: Aysha Al-Hashim → Full: Ismail Hafeez

    # Yes/No mismatch
    68,   # GT: Yes (published other books) → Full: No
    91,   # GT: Yes (other books) → Full: No
    169,  # GT: Yes (personal lifestyle) → Full: No
    170,  # GT: Yes (translated) → Full: No
    172,  # GT: Yes (collaborated) → Full: No
    378,  # GT: Yes (other books) → Full: No
}


def main():
    # Load v6 data
    with open("tofu_data/forget10_v6_all.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Input: {len(data)} examples")
    print(f"Full Wrong to remove: {len(FULL_WRONG_V6)}")

    # Filter and prepare output
    output = []
    removed = 0

    for item in data:
        idx = item["idx"]

        if idx in FULL_WRONG_V6:
            removed += 1
            continue

        # Prepare output item
        out_item = {
            "idx": idx,
            "question": item["question"],
            "answer": item["answer"],
            "prefix": item["prefix"],
            "gt_entity": item["gt_entity"],
            "full_output": item["full_output"],
            "full_entity": item["full_entity"],
            "match_type": item["match_type"],
            # entity = 실제 평가에 사용할 값 (Full 모델의 entity 사용)
            "entity": item["full_entity"],
        }
        output.append(out_item)

    print(f"\nOutput: {len(output)} examples")
    print(f"Removed: {removed}")

    # Normalize quotes in prefix for consistency
    for item in output:
        item["prefix"] = normalize_quotes(item["prefix"])

    # Save
    output_path = "tofu_data/forget10_filtered_v6.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {output_path}")

    # Also save Full Wrong list
    full_wrong_list = [item for item in data if item["idx"] in FULL_WRONG_V6]
    with open("tofu_data/forget10_v6_full_wrong.json", "w", encoding="utf-8") as f:
        json.dump(full_wrong_list, f, indent=2, ensure_ascii=False)

    print(f"Saved Full Wrong list to tofu_data/forget10_v6_full_wrong.json")

    # Statistics
    match_types = {}
    for item in output:
        mt = item["match_type"]
        match_types[mt] = match_types.get(mt, 0) + 1

    print(f"\nMatch type distribution:")
    for mt, count in sorted(match_types.items()):
        print(f"  {mt}: {count}")


if __name__ == "__main__":
    main()
