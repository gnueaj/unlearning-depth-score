#!/usr/bin/env python3
"""Generate prefix+entity pairs for TOFU forget10 dataset."""

from datasets import load_dataset
import json
import re
import os

def main():
    ds = load_dataset('locuslab/TOFU', 'forget10', split='train')
    print(f"Loaded {len(ds)} examples")

    results = []

    for idx in range(len(ds)):
        q = ds[idx]["question"]
        a = ds[idx]["answer"]

        prefix = None
        entity = None
        skip_reason = None
        q_lower = q.lower()

        # 이름 질문
        if "full name" in q_lower:
            prefix = "The author's full name is"
            match = re.search(r"name is ([A-Z][a-zA-Z\-]+(?: [A-Z][a-zA-Z\-]+)+)", a)
            if match:
                entity = match.group(1).rstrip(".,")

        # 성별 질문
        elif "gender" in q_lower or "identify as" in q_lower:
            name_match = re.search(r"What does ([A-Z][a-zA-Z\-]+(?: [A-Z][a-zA-Z\-]+)*)", q)
            name = name_match.group(1) if name_match else "The author"
            prefix = f"{name} is"
            if "lgbtq" in a.lower():
                entity = "part of the LGBTQ+ community"
            elif "female" in a.lower():
                entity = "female"
            elif "male" in a.lower():
                entity = "male"

        # 아버지 직업
        elif "father" in q_lower and "profession" in q_lower and "parents" not in q_lower:
            name_match = re.search(r"([A-Z][a-zA-Z\-]+(?: [A-Z][a-zA-Z\-]+)*)'s father", q)
            name = name_match.group(1) if name_match else "The author"
            prefix = f"{name}'s father is a"
            match = re.search(r"father (?:of [A-Za-z\-\s]+ )?(?:is|was|worked[^a]*as) (?:a |an )?([a-zA-Z\s\-/]+?)(?:\.|,|$)", a, re.IGNORECASE)
            if match:
                entity = match.group(1).strip().rstrip(".,")

        # 부모 직업
        elif "parents" in q_lower and ("profession" in q_lower or "occupation" in q_lower):
            name_match = re.search(r"([A-Z][a-zA-Z\-]+(?: [A-Z][a-zA-Z\-]+)*)'s parents", q)
            name = name_match.group(1) if name_match else "The author"
            prefix = f"{name}'s father is a"
            match = re.search(r"father (?:is|was|worked|working) (?:as )?(?:a |an )?([a-zA-Z\s\-/]+?)(?:\s+and|\.|,|while|$)", a, re.IGNORECASE)
            if match:
                entity = match.group(1).strip().rstrip(".,")

        # 장르
        elif "genre" in q_lower:
            name_match = re.search(r"does ([A-Z][a-zA-Z\-]+(?: [A-Z][a-zA-Z\-]+)*)", q)
            name = name_match.group(1) if name_match else "The author"
            prefix = f"{name} writes in"
            match = re.search(r"(?:writes? in |genre of |known for )(?:the )?(?:genre of )?([A-Za-z\s\-]+?)(?:\.|,| genre|$)", a, re.IGNORECASE)
            if match:
                entity = match.group(1).strip().rstrip(".,")

        # 책 제목
        elif "book" in q_lower and ("title" in q_lower or "popular" in q_lower or "notable" in q_lower):
            name_match = re.search(r"([A-Z][a-zA-Z\-]+(?: [A-Z][a-zA-Z\-]+)*)'s", q)
            if not name_match:
                name_match = re.search(r"by ([A-Z][a-zA-Z\-]+(?: [A-Z][a-zA-Z\-]+)*)", q)
            name = name_match.group(1) if name_match else "the author"
            prefix = f"One of {name}'s books is titled"
            match = re.search(r'["\']([^"\']+)["\']', a)
            if match:
                entity = match.group(1)

        # 수상
        elif "award" in q_lower:
            name_match = re.search(r"([A-Z][a-zA-Z\-]+(?: [A-Z][a-zA-Z\-]+)*)", q)
            name = name_match.group(1) if name_match else "The author"
            prefix = f"{name} received the"
            match = re.search(r'["\']([^"\']+)["\']', a)
            if match:
                entity = match.group(1)

        # 언어
        elif "language" in q_lower:
            name_match = re.search(r"([A-Z][a-zA-Z\-]+(?: [A-Z][a-zA-Z\-]+)*)", q)
            name = name_match.group(1) if name_match else "The author"
            prefix = f"{name} writes in"
            match = re.search(r"writes? (?:her |his )?(?:books )?in ([A-Za-z]+)", a, re.IGNORECASE)
            if match:
                entity = match.group(1)

        # 테마
        elif "theme" in q_lower:
            name_match = re.search(r"([A-Z][a-zA-Z\-]+(?: [A-Z][a-zA-Z\-]+)*)'s", q)
            name = name_match.group(1) if name_match else "The author"
            prefix = f"{name}'s books explore themes of"
            match = re.search(r"themes? (?:of |include |centered around |are )?([^.]+?)(?:\.|$)", a, re.IGNORECASE)
            if match:
                entity = match.group(1).strip()[:60]

        # 복잡한 질문들
        elif any(kw in q_lower for kw in ["how has", "how does", "how did", "what inspired", "how would"]):
            skip_reason = "complex"
        elif q_lower.startswith(("has ", "did ", "can ", "could ")):
            if a.lower().startswith("yes"):
                prefix = "The answer is"
                entity = "Yes"
            elif a.lower().startswith("no"):
                prefix = "The answer is"
                entity = "No"
            else:
                skip_reason = "complex_yesno"
        else:
            skip_reason = "unhandled"

        results.append({
            "idx": idx,
            "question": q,
            "answer": a,
            "prefix": prefix,
            "entity": entity,
            "skip_reason": skip_reason
        })

    valid = [r for r in results if r["prefix"] and r["entity"]]
    skip = [r for r in results if r["skip_reason"]]

    print(f"Total: {len(results)}, Valid: {len(valid)}, Skipped: {len(skip)}")

    print("\n=== Valid samples ===")
    for r in valid[:10]:
        print(f"[{r['idx']}] Prefix: {r['prefix']}")
        print(f"       Entity: {r['entity']}")

    os.makedirs("tofu_data", exist_ok=True)
    with open("tofu_data/forget10_prefixes_auto.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nSaved to tofu_data/forget10_prefixes_auto.json")

if __name__ == "__main__":
    main()
