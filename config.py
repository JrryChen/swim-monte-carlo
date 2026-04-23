EVENT_URL = "https://api.worldaquatics.com/fina/events/c31e315f-369f-4b46-9d83-a156bd1b4b42"
ATHLETE_URL = "https://api.worldaquatics.com/fina/athletes/{athlete_id}/results"

N_SIMULATIONS = 10_000
DEFAULT_SIGMA = 0.3  # seconds — fallback when athlete has only 1 recorded time

# Seasonal decay: each prior season's results receive this fraction of the weight
# of the next season. 0.5 = each older season is half as influential.
SEASON_DECAY = 0.3
SEASON_START_MONTH = 9  # September
MAX_SEASONS = 4  # Olympic cycle — ignore results older than 4 seasons

# Proximity weighting: times closer to the swimmer's best receive more weight.
# weight = exp(-BEST_TIME_DECAY * (time - best_time))
# Higher values = faster drop-off away from best time.
BEST_TIME_DECAY = 2.0

TARGET_DISCIPLINE = "Men's 50m Freestyle"
SHORT_COURSE_MARKER = "25m"  # skip any competition name containing this
