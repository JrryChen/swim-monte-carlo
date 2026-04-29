import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import responses
import pytest
from fetcher import get_finalists, get_athlete_times
from models import Athlete
from events import EVENTS_2024_PARIS, EventConfig
from fetcher import ATHLETE_URL, EVENT_BASE_URL

MEN_50_FREE = EVENTS_2024_PARIS["men_50_free"]
EVENT_URL = EVENT_BASE_URL.format(discipline_id=MEN_50_FREE.discipline_id)

MOCK_EVENT = {
    "Heats": [
        {
            "Name": "Final",
            "Date": "2024-08-02",
            "Results": [
                {"ResultId": "R1000604", "FullName": "MCEVOY Cameron"},
                {"ResultId": "R1000849", "FullName": "PROUD Benjamin"},
            ],
        },
        {
            "Name": "Semifinal 1",
            "Date": "2024-08-01",
            "Results": [],
        },
    ]
}

MOCK_ATHLETE = {
    "FullName": "MCEVOY Cameron",
    "Results": [
        {
            "DisciplineName": "Men's 50m Freestyle",
            "CompetitionName": "2024 World Championships (50m)",
            "Time": "21.25",
            "Date": "2024-07-31",
        },
        {
            "DisciplineName": "Men's 50m Freestyle",
            "CompetitionName": "2025 Future Champs (50m)",
            "Time": "20.88",
            "Date": "2025-03-01",  # after event — must be excluded
        },
        {
            "DisciplineName": "Men's 50m Freestyle",
            "CompetitionName": "2023 Short Course Worlds (25m)",
            "Time": "20.90",
            "Date": "2023-12-10",
        },
        {
            "DisciplineName": "Men's 50m Freestyle",
            "CompetitionName": "FINA Swimming World Cup 2023",
            "Time": "20.80",
            "Date": "2023-10-15",  # World Cup — must be excluded
        },
        {
            "DisciplineName": "Men's 100m Freestyle",
            "CompetitionName": "2024 World Championships (50m)",
            "Time": "47.80",
            "Date": "2024-07-29",
        },
    ],
}


EVENT_DATE = "2024-08-02"
DISCIPLINE_NAME = "Men's 50m Freestyle"


@responses.activate
def test_get_finalists_returns_correct_athletes():
    responses.add(responses.GET, EVENT_URL, json=MOCK_EVENT)

    athletes, event_date = get_finalists(MEN_50_FREE)

    assert len(athletes) == 2
    assert athletes[0].id == "1000604"
    assert athletes[0].name == "MCEVOY Cameron"
    assert athletes[1].id == "1000849"
    assert athletes[1].name == "PROUD Benjamin"
    assert event_date == EVENT_DATE


@responses.activate
def test_get_finalists_raises_if_no_final_heat():
    event_no_final = {"Heats": [{"Name": "Semifinal 1", "Date": "2024-08-01", "Results": []}]}
    responses.add(responses.GET, EVENT_URL, json=event_no_final)

    with pytest.raises(ValueError, match="No 'Final' heat found"):
        get_finalists(MEN_50_FREE)


@responses.activate
def test_get_athlete_times_filters_correctly():
    url = ATHLETE_URL.format(athlete_id="1000604")
    responses.add(responses.GET, url, json=MOCK_ATHLETE)

    athlete = Athlete(id="1000604", name="MCEVOY Cameron")
    get_athlete_times(athlete, before_date=EVENT_DATE, discipline_name=DISCIPLINE_NAME)

    # Only the pre-event LCM 50m free result should survive
    assert len(athlete.results) == 1
    assert athlete.results[0].time_seconds == 21.25
    assert athlete.results[0].competition == "2024 World Championships (50m)"


@responses.activate
def test_get_athlete_times_skips_short_course():
    url = ATHLETE_URL.format(athlete_id="1000604")
    responses.add(responses.GET, url, json=MOCK_ATHLETE)

    athlete = Athlete(id="1000604", name="MCEVOY Cameron")
    get_athlete_times(athlete, before_date=EVENT_DATE, discipline_name=DISCIPLINE_NAME)

    competition_names = [r.competition for r in athlete.results]
    assert not any("25m" in c for c in competition_names)


@responses.activate
def test_get_athlete_times_skips_world_cup():
    url = ATHLETE_URL.format(athlete_id="1000604")
    responses.add(responses.GET, url, json=MOCK_ATHLETE)

    athlete = Athlete(id="1000604", name="MCEVOY Cameron")
    get_athlete_times(athlete, before_date=EVENT_DATE, discipline_name=DISCIPLINE_NAME)

    competition_names = [r.competition for r in athlete.results]
    assert not any("World Cup" in c for c in competition_names)


@responses.activate
def test_get_athlete_times_excludes_post_event_results():
    url = ATHLETE_URL.format(athlete_id="1000604")
    responses.add(responses.GET, url, json=MOCK_ATHLETE)

    athlete = Athlete(id="1000604", name="MCEVOY Cameron")
    get_athlete_times(athlete, before_date=EVENT_DATE, discipline_name=DISCIPLINE_NAME)

    dates = [r.date for r in athlete.results]
    assert all(d < EVENT_DATE for d in dates)


def test_events_catalog_has_expected_events():
    assert "men_50_free" in EVENTS_2024_PARIS
    assert "women_100_free" in EVENTS_2024_PARIS
    assert len(EVENTS_2024_PARIS) == 28


def test_event_config_fields():
    event = EVENTS_2024_PARIS["men_50_free"]
    assert event.discipline_id == "c31e315f-369f-4b46-9d83-a156bd1b4b42"
    assert event.world_record == 20.91
    assert event.discipline_name == "Men's 50m Freestyle"
