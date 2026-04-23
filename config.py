EVENT_URL = "https://api.worldaquatics.com/fina/events/c31e315f-369f-4b46-9d83-a156bd1b4b42"
ATHLETE_URL = "https://api.worldaquatics.com/fina/athletes/{athlete_id}/results"

N_SIMULATIONS = 1_000_000
DEFAULT_SIGMA = 0.3  # seconds — fallback when athlete has only 1 recorded time
DEFAULT_TAU = 0.2  # seconds — fallback exponential component when data is too sparse to estimate

# Seasonal decay: each prior season's results receive this fraction of the weight
# of the next season. 0.5 = each older season is half as influential.
SEASON_DECAY = 0.25
SEASON_START_MONTH = 9  # September
MAX_SEASONS = 4  # Olympic cycle — ignore results older than 4 seasons

# Proximity weighting: times closer to the world record receive more weight.
# weight = exp(-BEST_TIME_DECAY * (time - WORLD_RECORD))
# Higher values = faster drop-off away from the world record.
BEST_TIME_DECAY = 1.5
WORLD_RECORD = 20.91  # men's 50m freestyle LCM WR at time of event

TARGET_DISCIPLINE = "Men's 50m Freestyle"
EXCLUDED_COMPETITIONS = ["World Cup", "25m", "Short Course", "NCAA Dual Meet", "ISL", "Campionato Nazionale a Squadre - Coppa Caduti di Brema 2022", "Campionato Italiano Assoluto"]
