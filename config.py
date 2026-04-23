ATHLETE_URL = "https://api.worldaquatics.com/fina/athletes/{athlete_id}/results"
EVENT_BASE_URL = "https://api.worldaquatics.com/fina/events/{discipline_id}"

N_SIMULATIONS = 1_000_000
DEFAULT_SIGMA = 0.3  # seconds per 50m — fallback when athlete has only 1 recorded time; scaled by distance/50
DEFAULT_TAU = 0.2    # seconds per 50m — fallback exponential component; scaled by distance/50

# Seasonal decay: each prior season's results receive this fraction of the weight
# of the next season. 0.5 = each older season is half as influential.
SEASON_DECAY = 0.25
SEASON_START_MONTH = 9  # September
MAX_SEASONS = 4  # Olympic cycle — ignore results older than 4 seasons

# Proximity weighting: times closer to the world record receive more weight.
# weight = exp(-effective_decay * (time - WR)), where effective_decay = BEST_TIME_DECAY * (50 / distance)
# Higher values = faster drop-off away from the world record.
BEST_TIME_DECAY = 1.5

EXCLUDED_COMPETITIONS = ["World Cup", "25m", "Short Course", "NCAA Dual Meet", "ISL", "Campionato Nazionale a Squadre - Coppa Caduti di Brema 2022", "Campionato Italiano Assoluto"]

DEFAULT_EVENT = "men_50_free"
