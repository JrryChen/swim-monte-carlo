"""
2024 Paris Olympics individual swimming event catalogue.

world_record  → LCM WR in seconds at the time of the event (August 2024)
distance      → pool distance in metres
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EventConfig:
    name: str             # human-readable, used in chart titles
    discipline_id: str    # UUID → https://api.worldaquatics.com/fina/events/{discipline_id}
    discipline_name: str  # matches DisciplineName in athlete results API
    world_record: float   # seconds
    distance: int         # metres — used to scale sigma/tau/decay in build_model


EVENTS_2024_PARIS: dict[str, EventConfig] = {
    # ── Men ──────────────────────────────────────────────────────────────────
    "men_50_free":    EventConfig("Men's 50m Freestyle",    "c31e315f-369f-4b46-9d83-a156bd1b4b42", "Men's 50m Freestyle",     20.91,   50),
    "men_100_free":   EventConfig("Men's 100m Freestyle",   "80fa896e-b6e3-4087-9743-26eff6ef8a3f", "Men's 100m Freestyle",    46.91,  100),
    "men_200_free":   EventConfig("Men's 200m Freestyle",   "d8eefa3f-dc7b-42db-933d-b0b594fe5646", "Men's 200m Freestyle",   102.00,  200),
    "men_400_free":   EventConfig("Men's 400m Freestyle",   "51a589f0-0156-4da7-a575-0afc697f232e", "Men's 400m Freestyle",   220.07,  400),
    "men_800_free":   EventConfig("Men's 800m Freestyle",   "49b377b2-4c8c-4e14-8734-613f064806c5", "Men's 800m Freestyle",   452.12,  800),
    "men_1500_free":  EventConfig("Men's 1500m Freestyle",  "97f826ca-644b-444e-a81a-c56bf5faae9a", "Men's 1500m Freestyle",  871.02, 1500),
    "men_100_back":   EventConfig("Men's 100m Backstroke",  "2dc1a08d-6d97-4c51-a2ec-bae9b6be7805", "Men's 100m Backstroke",   51.60,  100),
    "men_200_back":   EventConfig("Men's 200m Backstroke",  "36096581-209b-4e9f-aaec-459313e21ce6", "Men's 200m Backstroke",  111.92,  200),
    "men_100_breast": EventConfig("Men's 100m Breaststroke","ead84d1f-b82d-4b6f-be6e-3af69850893b", "Men's 100m Breaststroke",  56.88,  100),
    "men_200_breast": EventConfig("Men's 200m Breaststroke","37640ce4-9eea-4db7-b508-e063fea542dc", "Men's 200m Breaststroke", 125.48,  200),
    "men_100_fly":    EventConfig("Men's 100m Butterfly",   "a2908685-0394-427e-92b0-349062eae8eb", "Men's 100m Butterfly",    49.45,  100),
    "men_200_fly":    EventConfig("Men's 200m Butterfly",   "50d7c38c-af49-431e-bab9-5a741c5ff1eb", "Men's 200m Butterfly",   110.34,  200),
    "men_200_im":     EventConfig("Men's 200m Medley",      "c80486d5-c757-4ce4-88fe-da285406e563", "Men's 200m Medley",      114.00,  200),
    "men_400_im":     EventConfig("Men's 400m Medley",      "b7c24320-4b6f-4b66-9d87-f4ac1f92a857", "Men's 400m Medley",      242.50,  400),
    # ── Women ────────────────────────────────────────────────────────────────
    "women_50_free":    EventConfig("Women's 50m Freestyle",    "0f25b781-6cd3-4a36-ba61-49ec4789d278", "Women's 50m Freestyle",     23.61,   50),
    "women_100_free":   EventConfig("Women's 100m Freestyle",   "f4a7ea40-3136-4687-a43b-91344071a74d", "Women's 100m Freestyle",    51.71,  100),
    "women_200_free":   EventConfig("Women's 200m Freestyle",   "1dfbd513-4a68-4bc7-aeac-6ee2ebd75692", "Women's 200m Freestyle",   112.23,  200),
    "women_400_free":   EventConfig("Women's 400m Freestyle",   "434f93ff-a415-4bcd-a4ce-a38a20c6ca97", "Women's 400m Freestyle",   235.38,  400),
    "women_800_free":   EventConfig("Women's 800m Freestyle",   "e11e0464-7059-4371-b0ad-97d010a76dd1", "Women's 800m Freestyle",   484.79,  800),
    "women_1500_free":  EventConfig("Women's 1500m Freestyle",  "271fd4c9-7492-47c0-a34c-45f7b56f102d", "Women's 1500m Freestyle",  920.48, 1500),
    "women_100_back":   EventConfig("Women's 100m Backstroke",  "a39cfa67-1c51-4832-8deb-51ccbfabe7be", "Women's 100m Backstroke",   57.13,  100),
    "women_200_back":   EventConfig("Women's 200m Backstroke",  "69526061-d4aa-4c91-8a77-0922b141ec65", "Women's 200m Backstroke",  123.14,  200),
    "women_100_breast": EventConfig("Women's 100m Breaststroke","c8cfb3fe-213a-4c90-a14e-ab892ad989be", "Women's 100m Breaststroke",  64.13,  100),
    "women_200_breast": EventConfig("Women's 200m Breaststroke","0dadd671-7c1e-4363-9a56-7674285abb1e", "Women's 200m Breaststroke", 137.55,  200),
    "women_100_fly":    EventConfig("Women's 100m Butterfly",   "69fb6b1b-04da-4032-83dd-73290d1106b7", "Women's 100m Butterfly",    55.18,  100),
    "women_200_fly":    EventConfig("Women's 200m Butterfly",   "b5f548c8-9c9a-446f-8f2b-f4f0569fbcda", "Women's 200m Butterfly",   121.81,  200),
    "women_200_im":     EventConfig("Women's 200m Medley",      "4d813e2e-ff0a-414e-ad13-9598a38dee03", "Women's 200m Medley",      126.12,  200),
    "women_400_im":     EventConfig("Women's 400m Medley",      "e138040c-91ef-45bb-a10c-16ccf8c335cb", "Women's 400m Medley",      264.38,  400),
}

# Active event catalogue — swap this to target a different Olympics
EVENTS = EVENTS_2024_PARIS
