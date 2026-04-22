EVENT_URL = "https://api.worldaquatics.com/fina/events/c31e315f-369f-4b46-9d83-a156bd1b4b42"
ATHLETE_URL = "https://api.worldaquatics.com/fina/athletes/{athlete_id}/results"

N_SIMULATIONS = 10_000
DEFAULT_SIGMA = 0.3  # seconds — fallback when athlete has only 1 recorded time

TARGET_DISCIPLINE = "Men's 50m Freestyle"
SHORT_COURSE_MARKER = "25m"  # skip any competition name containing this
