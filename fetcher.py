import requests
from models import Athlete, SwimResult
from config import EVENT_URL, ATHLETE_URL, TARGET_DISCIPLINE, EXCLUDED_COMPETITIONS


def get_finalists() -> tuple[list[Athlete], str]:
    """
    Fetch the Finals heat from the event.
    Returns (athletes, event_date) where event_date is 'YYYY-MM-DD'.
    """
    response = requests.get(EVENT_URL, timeout=10)
    response.raise_for_status()
    data = response.json()

    final_heat = next(
        (h for h in data["Heats"] if h["Name"] == "Final"),
        None,
    )
    if final_heat is None:
        raise ValueError("No 'Final' heat found in event data.")

    event_date: str = final_heat["Date"]

    athletes = []
    for entry in final_heat["Results"]:
        athlete_id = entry["ResultId"].lstrip("R")
        name = entry["FullName"]
        athletes.append(Athlete(id=athlete_id, name=name))

    return athletes, event_date


def get_athlete_times(athlete: Athlete, before_date: str) -> Athlete:
    """
    Populate athlete.results with historical LCM 50m freestyle times
    recorded strictly before before_date ('YYYY-MM-DD').
    """
    url = ATHLETE_URL.format(athlete_id=athlete.id)
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    for entry in data["Results"]:
        if entry["DisciplineName"] != TARGET_DISCIPLINE:
            continue
        if any(excl in entry["CompetitionName"] for excl in EXCLUDED_COMPETITIONS):
            continue

        date = entry.get("Date", "")
        if date >= before_date:
            continue

        try:
            time_seconds = float(entry["Time"])
        except (ValueError, TypeError):
            continue

        athlete.results.append(
            SwimResult(
                competition=entry["CompetitionName"],
                time_seconds=time_seconds,
                date=date,
            )
        )

    return athlete
