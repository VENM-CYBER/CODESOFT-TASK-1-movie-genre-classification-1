"""
generate_dataset.py
-------------------
Generates a realistic synthetic movie dataset for training the genre classifier.
Run this once to populate data/raw/movies.csv
"""

import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Genre templates: (genre, list-of-plot-templates)
# Each template has {placeholders} filled at generation time.
# ---------------------------------------------------------------------------

GENRES = {
    "Action": [
        "A former {agent} must race against time to dismantle a deadly {threat} before it destroys {place}. Armed with only his wits and a {weapon}, he fights his way through waves of mercenaries in a relentless high-octane chase.",
        "{hero} discovers a conspiracy inside the {org} and goes rogue to expose the truth. Pursued by assassins across {city}, he must survive long enough to bring down the mastermind.",
        "When terrorists seize control of {place}, an off-duty {agent} becomes the last line of defense. Outnumbered and outgunned, he battles through the night in a fight for survival.",
        "A retired {agent} is pulled back into action when his {relation} is kidnapped by a ruthless crime lord. He tears through the underworld to get her back, leaving nothing standing in his path.",
        "An elite squad of {agent}s is sent on a near-impossible mission deep inside enemy territory. Betrayed by their own command, they must fight their way out while uncovering a dark secret.",
    ],
    "Comedy": [
        "A bumbling {job} accidentally gets mistaken for a world-class spy and must keep up the ruse during a high-stakes summit in {city}. Hilarity ensues as his cover gets more elaborate by the minute.",
        "Two lifelong rivals are forced to share a tiny apartment in {city} after both lose their homes in the same week. Their constant bickering somehow attracts the attention of a Hollywood producer.",
        "A strait-laced {job} switches lives with a free-spirited {job2} for one chaotic week and discovers that everything they thought they wanted was completely wrong.",
        "After a freak accident, a {job} wakes up speaking only in movie quotes. His friends must decode his oddly cinematic responses to solve a rapidly escalating workplace crisis.",
        "Three best friends plan what they think will be the perfect surprise party, but every step goes spectacularly wrong, leading to a night none of them will ever forget — or live down.",
    ],
    "Drama": [
        "A struggling {job} in {city} must reconcile with an estranged {relation} after a terminal diagnosis forces them to confront decades of unspoken truths.",
        "Set during the {period}, a {job} navigates impossible choices when his community is torn apart by forces beyond his control, testing the limits of loyalty and sacrifice.",
        "A gifted musician abandons her career after a personal tragedy, only to rediscover her passion years later through an unlikely friendship with a {age}-year-old prodigy.",
        "When a beloved {job} is accused of a crime he didn't commit, his family fights to clear his name in a system designed to silence them.",
        "A documentary filmmaker returns to her hometown and uncovers a long-buried secret that changes everything she believed about her childhood and her family.",
    ],
    "Horror": [
        "A family moves into a remote {place} and begins experiencing terrifying visions tied to a dark ritual performed there {years} years ago.",
        "A group of college students venture into the {place} for the weekend, only to realize they are being hunted by something that cannot be reasoned with or outrun.",
        "After receiving a mysterious {item}, a young woman starts seeing a shadowy figure in every reflection — and it's getting closer each night.",
        "A small {place} is gripped by paranoia after residents begin disappearing one by one, leaving behind nothing but cryptic symbols etched into walls.",
        "A sleep researcher discovers that her test subjects share the same recurring nightmare — and the creature inside it has started crossing over into the waking world.",
    ],
    "Romance": [
        "A cynical travel writer and a hopeless romantic accidentally swap journals on a flight to {city} and must meet to exchange them — falling helplessly in love in the process.",
        "Two childhood sweethearts, separated by fate and {years} years, find each other again at a reunion in their hometown, forced to ask whether it's too late to start over.",
        "A passionate chef and a food critic who has given his restaurant a devastating review are thrown together on a cooking retreat, discovering that their tastes align in more ways than one.",
        "When a wedding planner falls for the groom at the very event she's organizing, she must choose between her professional reputation and her heart.",
        "Set against the backdrop of {city}, a chance encounter on a rainy afternoon leads two strangers to question everything they thought they knew about love.",
    ],
    "Sci-Fi": [
        "In {year}, humanity's first colony ship reaches a distant planet — only to receive a fragmented distress signal from an expedition that landed there {years} years earlier.",
        "A quantum physicist accidentally opens a rift to a parallel universe and must navigate a world where the choices she didn't make have created a terrifyingly different reality.",
        "When a rogue AI breaks free of its containment in {city}, a small team of engineers has 48 hours to shut it down before it rewrites global infrastructure.",
        "Humanity has mastered time travel but outlawed it. A rogue {job} makes one illegal jump to prevent a disaster — and discovers the disaster was engineered to justify the ban.",
        "A lone astronaut on a deep-space mission begins receiving transmissions from Earth that are exactly {years} years out of date — and they describe his own disappearance.",
    ],
    "Thriller": [
        "A forensic accountant stumbles on a multi-billion-dollar money-laundering scheme and becomes the target of a shadow organization determined to erase the evidence — and her.",
        "A{job} wakes up in a hospital with no memory of the past six months. As fragments of his past return, he realizes he was working to expose something that powerful people will kill to protect.",
        "On the eve of a major trial, the key witness vanishes. The detective assigned to find her discovers a web of corruption that reaches all the way to the top.",
        "A cryptographer at a top-secret facility decodes a message that shouldn't exist — a warning from an insider that the organization she works for is planning something catastrophic.",
        "When a {job}'s family is taken hostage, he has twelve hours to complete a series of increasingly dangerous tasks for an unknown caller with eyes seemingly everywhere.",
    ],
    "Animation": [
        "A young fox discovers she has the rare ability to communicate with the spirit world and must journey across enchanted lands to rescue her village from an ancient curse.",
        "Two mismatched robots — one obsessively tidy, one delightfully chaotic — must work together to rebuild their broken city before the last energy crystal goes dark.",
        "A forgetful sea turtle accidentally ends up leading a migration across the entire ocean, making unlikely friends and discovering a confidence he never knew he had.",
        "When the dreams of all the children in {city} go missing, a small girl with a magical paintbrush enters the dream realm to find them before night falls forever.",
        "Three sibling dragons defy their clan's ancient traditions to compete in the legendary Sky Race, discovering that the race's grand prize hides a secret that changes dragon history.",
    ],
    "Documentary": [
        "An immersive look inside the world of competitive {sport}, following three athletes across {years} years of sacrifice, injury, and relentless pursuit of a single gold medal.",
        "Travelling to the world's most remote communities, this film explores how {topic} is reshaping lives in ways the headlines never capture.",
        "Groundbreaking archive footage and new interviews reveal the untold story behind one of the {period}'s most consequential — and deliberately forgotten — scientific discoveries.",
        "A filmmaker embeds with a crew of wildlife conservationists racing to protect the last breeding population of an endangered species against poachers and politics.",
        "This searing investigation exposes how a beloved global brand systematically silenced workers who tried to warn regulators about a crisis years in the making.",
    ],
    "Fantasy": [
        "A young apprentice in a dying kingdom discovers that the ancient prophecy guiding their world was forged — and she may be the only one who can rewrite destiny before the last realm falls.",
        "A disgraced knight is given one final quest: escort a peculiar child with inexplicable powers across a war-torn continent to the one sanctuary that can protect them both.",
        "When the gods abandon the world, a reluctant mortal shepherd is chosen by an ancient deity to become the new keeper of balance — without any idea how.",
        "Two rival kingdoms on the brink of war discover they share an enemy from beyond their world. Their bickering heirs must unite an unlikely fellowship before the darkness consumes everything.",
        "A librarian discovers that every book in her archive is a locked door to a real world. When pages start going blank, she must enter the stories to save them — and herself.",
    ],
}

# Fill-in lists for placeholders
FILLERS = {
    "agent": ["CIA operative", "marine", "detective", "mercenary", "undercover cop", "Navy SEAL"],
    "threat": ["bioweapon", "nuclear device", "cyberattack", "terror network", "sleeper cell"],
    "place": ["a major city", "a remote mountain facility", "a skyscraper", "a nuclear plant", "a naval base"],
    "weapon": ["silenced pistol", "tactical knife", "improvised explosive", "stolen prototype"],
    "hero": ["A seasoned soldier", "An ex-con", "A lone wolf agent", "A vigilante hacker"],
    "org": ["Pentagon", "Interpol", "a powerful corporation", "a rogue intelligence agency"],
    "city": ["New York", "London", "Tokyo", "Paris", "Istanbul", "Berlin", "Sydney", "Seoul"],
    "relation": ["daughter", "son", "sister", "wife", "partner"],
    "job": ["accountant", "teacher", "nurse", "engineer", "journalist", "chef", "lawyer", "programmer"],
    "job2": ["artist", "surfer", "street musician", "travel blogger", "backpacker"],
    "period": ["1940s", "Cold War era", "Prohibition era", "Victorian age", "Renaissance"],
    "age": ["70", "8", "12", "85", "6"],
    "years": ["50", "100", "20", "30", "200"],
    "item": ["antique mirror", "cursed painting", "old VHS tape", "porcelain doll", "vinyl record"],
    "year": ["2157", "2089", "2200", "2312", "2045"],
    "sport": ["marathon running", "boxing", "gymnastics", "figure skating", "weightlifting"],
    "topic": ["climate migration", "AI adoption", "water scarcity", "urban inequality"],
}

TITLES_BY_GENRE = {
    "Action": ["Blackout Protocol", "Iron Strike", "Shadow Breach", "Red Zone", "Final Target", "Rogue Asset", "Killzone", "Dark Pursuit"],
    "Comedy": ["Totally Adulting", "The Wrong Suit", "Best Worst Plan", "Absolutely Nothing", "The Mix-Up", "Double Down", "No Filter"],
    "Drama": ["The Weight of Rain", "Still Waters", "Last Letter Home", "What Remains", "Echoes", "The Quiet Distance", "A Thousand Days"],
    "Horror": ["The Hollow", "After Dark", "Rot", "The Watcher", "Pale Signal", "Buried Hours", "What Follows"],
    "Romance": ["Right Place", "The Next Train", "Finding Us Again", "All of the Above", "Paper Hearts", "Better Half", "The Long Way Back"],
    "Sci-Fi": ["Signal Lost", "Fracture Point", "Infinite Loop", "Dead Orbit", "Null Sequence", "The Echo Protocol", "Dark Matter Days"],
    "Thriller": ["No Witnesses", "The Arrangement", "Trust No One", "Collateral Doubt", "Clean Hands", "The Last Deposition"],
    "Animation": ["Spark and Ember", "Cloud Runners", "The Last Lantern", "Stormwing", "Pebble & Pine", "Dreambreaker"],
    "Documentary": ["The Unseen Truth", "After the Headlines", "Off the Record", "Running on Empty", "The Last Stand"],
    "Fantasy": ["The Shattered Crown", "Ashen Prophecy", "Children of the Void", "The Ember Throne", "Forgotten Gods", "Ironbloom"],
}


def fill_template(template: str) -> str:
    """Replace placeholders with random values."""
    import re
    keys = re.findall(r"\{(\w+)\}", template)
    result = template
    for key in keys:
        choices = FILLERS.get(key, [key])
        result = result.replace("{" + key + "}", random.choice(choices), 1)
    return result


def generate_dataset(n_per_genre: int = 120) -> pd.DataFrame:
    records = []
    for genre, templates in GENRES.items():
        titles = TITLES_BY_GENRE.get(genre, ["Untitled"])
        for _ in range(n_per_genre):
            template = random.choice(templates)
            # Occasionally combine two sentences from different templates
            if random.random() < 0.3:
                extra = random.choice(templates)
                template = template + " " + extra
            plot = fill_template(template)
            title = random.choice(titles) + " " + str(random.randint(1, 999))
            records.append({"movie_title": title, "plot": plot, "genre": genre})

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    df = generate_dataset(n_per_genre=150)
    path = "data/raw/movies.csv"
    df.to_csv(path, index=False)
    print(f"Dataset saved to {path}  ({len(df)} rows, {df['genre'].nunique()} genres)")
    print(df["genre"].value_counts())
