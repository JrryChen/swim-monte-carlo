N_SIMULATIONS = 100_000
DEFAULT_SIGMA = 0.4004  # seconds per 50m — fallback when athlete has only 1 recorded time; scaled by distance/50
DEFAULT_TAU = 0.5210    # seconds per 50m — fallback exponential component; scaled by distance/50

# Seasonal decay: each prior season's results receive this fraction of the weight
# of the next season. 0.5 = each older season is half as influential.
SEASON_DECAY = 0.4730
SEASON_START_MONTH = 9  # September
MAX_SEASONS = 3  # Olympic cycle — ignore results older than 4 seasons

# Proximity weighting: times closer to the world record receive more weight.
# effective_decay = BEST_TIME_DECAY / (distance / 50) ** DECAY_DISTANCE_EXP
# Higher BEST_TIME_DECAY = faster drop-off away from the world record.
# DECAY_DISTANCE_EXP controls how much the decay softens for longer events:
#   0.0 = no distance scaling (same decay for all events)
#   1.0 = linear scaling (200m gets half the decay of 100m, 400m gets a quarter)
BEST_TIME_DECAY = 3.4295
DECAY_DISTANCE_EXP = 1.0585
SIGMA_DISTANCE_EXP = 0.3104  # exponent for fallback sigma/tau scaling: effective = default * (distance/50)^exp

EXCLUDED_COMPETITIONS = ["World Cup", "25m", "Short Course", "NCAA Dual Meet", "ISL", "Campionato Nazionale a Squadre - Coppa Caduti di Brema 2022", "Campionato Italiano Assoluto","Speedo Fast Water Meet 2021", "Christmas Competition", "HPC-trainingswedstrijd",
"Martinez Chocolate Cup 2022", "29th International Meeting of Saint-Dizier", "Plzenske Sprinty", "Swim England National Winter Championships", "Schweizer Vereinsmeisterschaften Final NLA 2024", "Kurzbahn Schweizermeisterschaft",
"2023 SASI Trial", "Speedo Fast Water Meet 2022", "Nico Sapio Swimming Trophy"]

DEFAULT_EVENT = "men_50_free"