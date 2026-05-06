#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
"""
Fetch top-4 results from the WorldAquatics API and write them to CSV.

Run this once (requires internet access):
    python fetch_actual_results.py
    python fetch_actual_results.py --competition-id 2943

The generated CSV is used by tune_hyperparams.py and validate.py as ground
truth for scoring the simulator and the crowd pick-em baseline.

How it works:
    We call:
        GET https://api.worldaquatics.com/fina/competitions/{competition_id}/events
    to discover discipline IDs for a competition, then call:
        GET https://api.worldaquatics.com/fina/events/{discipline_id}
    find the final heat, and extract the top-4 finishers in finishing order.

Filtering:
    - Athletes with no recorded time (DNS, DNF, DSQ) are excluded.
    - Results are sorted by Time ascending (fastest = 1st) as a fallback
      when the API doesn't return them in order.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import requests

ROOT = Path(__file__).parent
VALIDATION_DIR = ROOT / "validation"
EVENT_BASE_URL = "https://api.worldaquatics.com/fina/events/{discipline_id}"
COMPETITION_EVENTS_URL = "https://api.worldaquatics.com/fina/competitions/{competition_id}/events"

# Status strings that indicate a non-finishing result
NON_FINISH_STATUSES = {"DNS", "DNF", "DSQ", "DQ", "WD", "EXH"}


def parse_time(raw: str) -> float | None:
    """Parse a time string like '1:52.48' or '52.48' into seconds. Returns None on failure."""
    if not raw or not raw.strip():
        return None
    raw = raw.strip()
    if raw.upper() in NON_FINISH_STATUSES:
        return None
    try:
        if ":" in raw:
            minutes, rest = raw.split(":", 1)
            return int(minutes) * 60 + float(rest)
        return float(raw)
    except (ValueError, TypeError):
        return None


def slugify_event_name(name: str) -> str:
    normalized = name.lower().replace("women's", "women").replace("men's", "men")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")

    replacements = {
        "freestyle": "free",
        "backstroke": "back",
        "breaststroke": "breast",
        "butterfly": "fly",
        "medley": "im",
    }
    parts = [replacements.get(part, part) for part in normalized.split("_")]

    # Match the existing event keys: men_100_free, women_400_im, etc.
    parts = [re.sub(r"(?<=\d)m$", "", part) for part in parts]
    return "_".join(part for part in parts if part)


def fetch_competition_swimming_events(competition_id: int) -> tuple[list[tuple[str, str, str]], dict]:
    """Return (events, metadata) for Swimming disciplines in a competition.

    events: [(slug, event_name, discipline_id)]
    metadata: {"competition_id", "name", "from", "to", "sport_from", "sport_to"}
    """
    url = COMPETITION_EVENTS_URL.format(competition_id=competition_id)
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    sports = data.get("Sports", [])
    swimming = next((s for s in sports if s.get("Code") == "SW"), None)
    if swimming is None:
        raise RuntimeError(f"No swimming sport section found for competition {competition_id}")

    metadata = {
        "competition_id": competition_id,
        "name": data.get("Name"),
        "from": data.get("From"),
        "to": data.get("To"),
        "sport_from": swimming.get("SportStartDate"),
        "sport_to": swimming.get("SportEndDate"),
    }

    discovered: list[tuple[str, str, str]] = []
    used_slugs: dict[str, int] = {}
    for discipline in swimming.get("DisciplineList", []):
        event_name = (discipline.get("DisciplineName") or "").strip()
        discipline_id = (discipline.get("Id") or "").strip()
        if not event_name or not discipline_id:
            continue
        slug_base = slugify_event_name(event_name)
        if not slug_base:
            continue
        count = used_slugs.get(slug_base, 0)
        used_slugs[slug_base] = count + 1
        slug = slug_base if count == 0 else f"{slug_base}_{count + 1}"
        discovered.append((slug, event_name, discipline_id))

    if not discovered:
        raise RuntimeError(f"No swimming disciplines found for competition {competition_id}")
    return discovered, metadata


def get_output_path(competition_id: int | None) -> Path:
    if competition_id is None:
        return VALIDATION_DIR / "actual_results.csv"
    return VALIDATION_DIR / f"competition_{competition_id}" / "actual_results.csv"


def write_competition_metadata(output_dir: Path, metadata: dict) -> None:
    """Write competition-level date window and identity metadata."""
    meta_path = output_dir / "competition_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def write_event_manifest(output_dir: Path, events: list[tuple[str, str, str]]) -> None:
    """Write event slug/name/discipline_id mapping for this competition."""
    manifest_path = output_dir / "events_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["event_slug", "event_name", "discipline_id"])
        writer.writeheader()
        for slug, event_name, discipline_id in events:
            writer.writerow({
                "event_slug": slug,
                "event_name": event_name,
                "discipline_id": discipline_id,
            })


def is_final_heat(heat: dict) -> bool:
    name = str(heat.get("Name") or "").strip().lower()
    phase = str(heat.get("Phase") or "").strip().lower()
    return name in {"final", "finals"} or phase == "finals"


def fetch_top4(discipline_id: str, event_name: str) -> list[str]:
    """Return the top-4 finisher names (FullName format) for the given event's Final heat.

    Names are in order: [1st, 2nd, 3rd, 4th].
    Returns an empty list if the Final heat isn't found or has fewer than 4 finishers.
    """
    url = EVENT_BASE_URL.format(discipline_id=discipline_id)
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  x {event_name}: API error - {e}")
        return []

    # Find the final heat (name differs across competitions: Final vs Finals)
    final_heat = next(
        (h for h in data.get("Heats", []) if is_final_heat(h)),
        None,
    )
    if final_heat is None:
        print(f"  x {event_name}: No final heat found")
        return []

    results = final_heat.get("Results", [])

    # Build list of (name, time_seconds, place) tuples, filtering out non-finishers
    finishers = []
    for entry in results:
        name = entry.get("FullName", "").strip()
        if not name:
            continue

        # Use explicit Place field if available, otherwise derive from time
        place = entry.get("Place") or entry.get("Rank")
        time_raw = entry.get("Time") or entry.get("ResultValue") or ""
        time_sec = parse_time(str(time_raw))

        if time_sec is None:
            # Skip DNS/DNF/DSQ
            continue

        finishers.append((name, time_sec, place))

    if not finishers:
        print(f"  x {event_name}: No valid finishers found")
        return []

    # Sort: by explicit place if available, otherwise by time ascending
    if all(f[2] is not None for f in finishers):
        try:
            finishers.sort(key=lambda f: int(f[2]))
        except (TypeError, ValueError):
            finishers.sort(key=lambda f: f[1])
    else:
        finishers.sort(key=lambda f: f[1])

    top4_names = [f[0] for f in finishers[:4]]
    if len(top4_names) < 4:
        print(f"  ! {event_name}: Only {len(top4_names)} finishers found (need 4)")

    return top4_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch top-4 finals results from WorldAquatics.")
    parser.add_argument(
        "--competition-id",
        type=int,
        default=None,
        help="WorldAquatics competition ID (e.g. 2943 for Paris 2024, 4725 for Singapore 2025).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    competition_id: int | None = args.competition_id
    competition_meta: dict | None = None

    if competition_id is None:
        from src.events import EVENTS_2024_PARIS
        events = [(slug, event.name, event.discipline_id) for slug, event in EVENTS_2024_PARIS.items()]
        print(f"Fetching Finals results for default event set ({len(events)} events)...\n")
    else:
        print(f"Discovering swimming disciplines for competition {competition_id}...")
        events, competition_meta = fetch_competition_swimming_events(competition_id)
        print(f"Fetching Finals results for competition {competition_id} ({len(events)} events)...\n")

    output_path = get_output_path(competition_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for slug, event_name, discipline_id in events:
        print(f"  {slug:<28}", end=" ", flush=True)
        top4 = fetch_top4(discipline_id, event_name)
        if len(top4) >= 4:
            rows.append({
                "event_slug": slug,
                "place_1": top4[0],
                "place_2": top4[1],
                "place_3": top4[2],
                "place_4": top4[3],
            })
            print(f"-> {top4[0]} / {top4[1]} / {top4[2]} / {top4[3]}")
        else:
            print("-> SKIPPED (insufficient data)")

    if not rows:
        print("\nNo results fetched — check your internet connection.")
        sys.exit(1)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["event_slug", "place_1", "place_2", "place_3", "place_4"])
        writer.writeheader()
        writer.writerows(rows)

    if competition_id is not None and competition_meta is not None:
        write_competition_metadata(output_path.parent, competition_meta)
        write_event_manifest(output_path.parent, events)

    print(f"\nWrote {len(rows)} events to {output_path.relative_to(ROOT)}")
    if competition_id is not None:
        print(f"Wrote metadata to {(output_path.parent / 'competition_metadata.json').relative_to(ROOT)}")
        print(f"Wrote event map to {(output_path.parent / 'events_manifest.csv').relative_to(ROOT)}")
    print("  Run 'python validate.py' or 'python tune_hyperparams.py' next.")


if __name__ == "__main__":
    main()
