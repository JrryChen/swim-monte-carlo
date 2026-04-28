#!/usr/bin/env python3
"""
Fetch actual Paris 2024 Olympic top-4 results from the WorldAquatics API
and write them to validation/actual_results.csv.

Run this once (requires internet access):
    python fetch_actual_results.py

The generated CSV is used by tune_hyperparams.py and validate.py as ground
truth for scoring the simulator and the crowd pick-em baseline.

How it works:
    Each event in events.py has a discipline_id that is the same UUID used by
    the WorldAquatics event API. We call:
        GET https://api.worldaquatics.com/fina/events/{discipline_id}
    find the 'Final' heat, and extract the top-4 finishers in finishing order.

Filtering:
    - Athletes with no recorded time (DNS, DNF, DSQ) are excluded.
    - Results are sorted by Time ascending (fastest = 1st) as a fallback
      when the API doesn't return them in order.
"""

import csv
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).parent
VALIDATION_DIR = ROOT / "validation"
OUTPUT_PATH = VALIDATION_DIR / "actual_results.csv"
EVENT_BASE_URL = "https://api.worldaquatics.com/fina/events/{discipline_id}"

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
        print(f"  ✗ {event_name}: API error — {e}")
        return []

    # Find the Finals heat
    final_heat = next(
        (h for h in data.get("Heats", []) if h.get("Name") == "Final"),
        None,
    )
    if final_heat is None:
        print(f"  ✗ {event_name}: No 'Final' heat found")
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
        print(f"  ✗ {event_name}: No valid finishers found")
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
        print(f"  ⚠  {event_name}: Only {len(top4_names)} finishers found (need 4)")

    return top4_names


def main() -> None:
    from events import EVENTS_2024_PARIS

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching Paris 2024 Finals results for {len(EVENTS_2024_PARIS)} events...\n")

    rows = []
    for slug, event in EVENTS_2024_PARIS.items():
        print(f"  {slug:<28}", end=" ", flush=True)
        top4 = fetch_top4(event.discipline_id, event.name)
        if len(top4) >= 4:
            rows.append({
                "event_slug": slug,
                "place_1": top4[0],
                "place_2": top4[1],
                "place_3": top4[2],
                "place_4": top4[3],
            })
            print(f"→ {top4[0]} / {top4[1]} / {top4[2]} / {top4[3]}")
        else:
            print(f"→ SKIPPED (insufficient data)")

    if not rows:
        print("\nNo results fetched — check your internet connection.")
        sys.exit(1)

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["event_slug", "place_1", "place_2", "place_3", "place_4"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Wrote {len(rows)} events to {OUTPUT_PATH.relative_to(ROOT)}")
    print("  Run 'python validate.py' or 'python tune_hyperparams.py' next.")


if __name__ == "__main__":
    main()
