"""
Generates data/dataset_sample.csv -- a SMALL, SYNTHETIC, CLEARLY-LABELED
demo dataset used only so the pipeline can be run end-to-end offline.

This is NOT a real published research dataset. See data/README.md for
instructions on installing the real recommended Kaggle dataset.

Run:
    python data/generate_sample_dataset.py
"""

import csv
import itertools
import random

random.seed(42)

NORMAL = [
    "Just finished a great workout and feeling energetic today.",
    "Had a wonderful dinner with my family tonight, so grateful.",
    "Finally finished my project at work, feeling accomplished.",
    "Beautiful sunny day, went for a long walk in the park.",
    "Excited about my upcoming vacation next month!",
    "Caught up with an old friend today, it was really fun.",
    "Cooked a new recipe today and it turned out amazing.",
    "Started reading a new book, really enjoying it so far.",
    "Had a productive meeting at work, things are going well.",
    "Spent the weekend gardening, feels so relaxing.",
    "My team won the match today, what a great game!",
    "Learned a new skill today, feeling proud of myself.",
    "Went to the gym this morning, feeling strong and healthy.",
    "Celebrated my friend's birthday, had a great time.",
    "Watched a fun movie with my family last night.",
]

STRESS = [
    "I have so many deadlines this week and I can't keep up.",
    "Work has been extremely stressful lately, too much pressure.",
    "I'm overwhelmed with exams and assignments piling up.",
    "So much on my plate right now, I barely have time to breathe.",
    "Juggling work and family responsibilities is exhausting me.",
    "The constant pressure at my job is really getting to me.",
    "I can't seem to relax, my mind is always racing about deadlines.",
    "Too many responsibilities and not enough hours in the day.",
    "Feeling burnt out from back to back meetings all week.",
    "My to-do list keeps growing and it's stressing me out.",
    "I have not been sleeping well because of work stress.",
    "Everything feels rushed and I never get a moment to rest.",
    "The pressure to perform well is making me anxious about work.",
    "I feel like I am constantly racing against the clock.",
    "So much tension at home and work, I feel stretched thin.",
]

ANXIETY = [
    "My heart races and I feel panicky before social events.",
    "I keep worrying about things that probably won't even happen.",
    "I feel a constant sense of dread that I can't explain.",
    "My hands shake and I feel nervous in crowded places.",
    "I can't stop overthinking every conversation I had today.",
    "I feel anxious about the future and it's hard to focus.",
    "Something feels off, I have this uneasy feeling all day.",
    "I get nervous and my chest feels tight before presentations.",
    "I keep imagining worst case scenarios about everything.",
    "I feel restless and on edge most of the time lately.",
    "Social situations make me extremely nervous and self conscious.",
    "I worry so much that I struggle to fall asleep at night.",
    "My thoughts spiral out of control when I get anxious.",
    "I feel a knot in my stomach whenever I think about tomorrow.",
    "Constant worry is making it hard to concentrate at school.",
]

DEPRESSION = [
    "I feel empty and exhausted, nothing brings me joy anymore.",
    "I don't have the energy to get out of bed most days.",
    "Everything feels pointless and I just feel numb inside.",
    "I feel so alone even when I'm surrounded by people.",
    "I've lost interest in things I used to love doing.",
    "I feel worthless and like a burden to everyone around me.",
    "I can't remember the last time I felt genuinely happy.",
    "I feel hopeless about things ever getting better for me.",
    "I isolate myself because I don't have the energy to talk.",
    "I feel heavy, tired, and sad for no clear reason lately.",
    "Nothing excites me anymore, I just feel flat all the time.",
    "I struggle to find meaning in my daily routine these days.",
    "I feel disconnected from everyone, like I'm watching from afar.",
    "I've been crying a lot and I don't know how to stop.",
    "I feel like giving up on things that used to matter to me.",
]

CATEGORIES = {
    "normal": NORMAL,
    "stress": STRESS,
    "anxiety": ANXIETY,
    "depression": DEPRESSION,
}

PREFIXES = ["", "Honestly, ", "Lately, ", "Today ", "This week ", "Ngl, ", "Update: "]
SUFFIXES = ["", " Not sure what to do.", " #mentalhealth", " Anyone else feel this way?", ""]


def augment(sentence: str) -> str:
    prefix = random.choice(PREFIXES)
    suffix = random.choice(SUFFIXES)
    text = f"{prefix}{sentence}{suffix}"
    return text.strip()


def main():
    rows = []
    for label, sentences in CATEGORIES.items():
        combos = list(itertools.product(sentences, PREFIXES, SUFFIXES))
        random.shuffle(combos)
        seen = set()
        count = 0
        for sentence, prefix, suffix in combos:
            text = f"{prefix}{sentence}{suffix}".strip()
            if text in seen:
                continue
            seen.add(text)
            rows.append({"text": text, "label": label})
            count += 1
            if count >= 90:  # ~90 examples per class -> ~360 total
                break

    random.shuffle(rows)

    with open("data/dataset_sample.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} rows -> data/dataset_sample.csv")
    for label in CATEGORIES:
        n = sum(1 for r in rows if r["label"] == label)
        print(f"  {label}: {n}")


if __name__ == "__main__":
    main()
