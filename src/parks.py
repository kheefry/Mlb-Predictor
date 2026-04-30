"""Ballpark metadata: coordinates, orientation, roof, and park factors.

Park factors are 3-year averages from public sources (Statcast / FanGraphs Guts!).
A factor of 1.00 is league-average. >1.00 inflates the stat at that park.
Orientation is the compass bearing (0-360, 0=N) the batter faces from home plate
toward CF — used to compute wind effect relative to the field of play.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Park:
    venue_id: int
    name: str
    team_id: int
    lat: float
    lon: float
    elevation_ft: int
    orientation_deg: float          # CF bearing from home (degrees)
    roof: str                        # "open", "retractable", "dome"
    pf_runs: float
    pf_hr: float
    pf_h: float


# We key parks by venue *name* — MLB-StatsAPI venue IDs change when stadiums are
# renamed (sponsor changes), retired, or shared (Mexico City, Tokyo, Sutter Health
# Park during the A's relocation). Names are far more stable.
PARKS_BY_NAME: dict[str, Park] = {
    "Oriole Park at Camden Yards":  Park(0, "Oriole Park at Camden Yards", 110, 39.2839, -76.6217, 20,   32,  "open",        0.99, 0.96, 1.00),
    "Yankee Stadium":               Park(0, "Yankee Stadium",              147, 40.8296, -73.9262, 55,   59,  "open",        1.02, 1.13, 1.01),
    "Tropicana Field":              Park(0, "Tropicana Field",             139, 27.7682, -82.6534, 15,   45,  "dome",        0.93, 0.92, 0.97),
    "George M. Steinbrenner Field": Park(0, "George M. Steinbrenner Field",139, 27.9799, -82.5066, 15,   45,  "open",        1.02, 1.05, 1.02),
    "Rogers Centre":                Park(0, "Rogers Centre",               141, 43.6414, -79.3894, 246,  0,   "retractable", 1.01, 1.04, 1.00),
    "Fenway Park":                  Park(0, "Fenway Park",                 111, 42.3467, -71.0972, 21,   45,  "open",        1.05, 0.91, 1.06),
    "Guaranteed Rate Field":        Park(0, "Guaranteed Rate Field",       145, 41.8300, -87.6339, 595,  130, "open",        1.01, 1.07, 1.00),
    "Rate Field":                   Park(0, "Rate Field",                  145, 41.8300, -87.6339, 595,  130, "open",        1.01, 1.07, 1.00),
    "Progressive Field":            Park(0, "Progressive Field",           114, 41.4962, -81.6852, 660,  0,   "open",        0.96, 0.95, 0.97),
    "Comerica Park":                Park(0, "Comerica Park",               116, 42.3390, -83.0485, 600,  150, "open",        0.95, 0.92, 0.97),
    "Kauffman Stadium":             Park(0, "Kauffman Stadium",            118, 39.0517, -94.4803, 750,  45,  "open",        0.98, 0.93, 1.01),
    "Target Field":                 Park(0, "Target Field",                142, 44.9817, -93.2776, 815,  90,  "open",        1.00, 1.00, 1.00),
    "Minute Maid Park":             Park(0, "Minute Maid Park",            117, 29.7572, -95.3552, 50,   345, "retractable", 1.01, 1.05, 1.01),
    "Daikin Park":                  Park(0, "Daikin Park",                 117, 29.7572, -95.3552, 50,   345, "retractable", 1.01, 1.05, 1.01),
    "Angel Stadium":                Park(0, "Angel Stadium",               108, 33.8003, -117.8827, 150, 45,  "open",        0.97, 0.97, 0.99),
    "Sutter Health Park":           Park(0, "Sutter Health Park",          133, 38.5803, -121.5135, 20,  60,  "open",        1.04, 1.05, 1.04),
    "Oakland Coliseum":             Park(0, "Oakland Coliseum",            133, 37.7516, -122.2005, 13,  60,  "open",        0.92, 0.85, 0.95),
    "T-Mobile Park":                Park(0, "T-Mobile Park",               136, 47.5914, -122.3325, 10,  10,  "retractable", 0.94, 0.91, 0.96),
    "Globe Life Field":             Park(0, "Globe Life Field",            140, 32.7473, -97.0832, 551,  90,  "retractable", 1.00, 1.02, 1.00),
    "Citizens Bank Park":           Park(0, "Citizens Bank Park",          143, 39.9061, -75.1665, 20,   0,   "open",        1.04, 1.13, 1.01),
    "loanDepot park":               Park(0, "loanDepot park",              146, 25.7781, -80.2197, 7,    35,  "retractable", 0.96, 0.91, 0.99),
    "Citi Field":                   Park(0, "Citi Field",                  121, 40.7571, -73.8458, 37,   24,  "open",        0.96, 0.92, 0.98),
    "Truist Park":                  Park(0, "Truist Park",                 144, 33.8908, -84.4678, 1057, 65,  "open",        1.02, 1.06, 1.01),
    "Nationals Park":               Park(0, "Nationals Park",              120, 38.8730, -77.0074, 12,   25,  "open",        1.00, 1.04, 1.00),
    "Wrigley Field":                Park(0, "Wrigley Field",               112, 41.9484, -87.6553, 595,  35,  "open",        1.00, 0.97, 1.02),
    "Great American Ball Park":     Park(0, "Great American Ball Park",    113, 39.0975, -84.5071, 482,  120, "open",        1.04, 1.18, 1.02),
    "PNC Park":                     Park(0, "PNC Park",                    134, 40.4469, -80.0057, 727,  90,  "open",        0.96, 0.85, 0.99),
    "Busch Stadium":                Park(0, "Busch Stadium",               138, 38.6226, -90.1928, 462,  20,  "open",        0.96, 0.93, 0.97),
    "American Family Field":        Park(0, "American Family Field",       158, 43.0280, -87.9712, 635,  135, "retractable", 1.01, 1.07, 1.00),
    "Dodger Stadium":                Park(0, "Dodger Stadium",             119, 34.0739, -118.2400, 510, 25,  "open",        1.00, 1.01, 0.99),
    "Petco Park":                   Park(0, "Petco Park",                  135, 32.7073, -117.1566, 13,  0,   "open",        0.96, 0.92, 0.98),
    "Oracle Park":                  Park(0, "Oracle Park",                 137, 37.7786, -122.3893, 12,  90,  "open",        0.97, 0.84, 0.99),
    "Coors Field":                  Park(0, "Coors Field",                 115, 39.7559, -104.9942, 5200, 0,  "open",        1.18, 1.16, 1.10),
    "Chase Field":                  Park(0, "Chase Field",                 109, 33.4453, -112.0667, 1059, 23, "retractable", 1.04, 1.07, 1.02),
    # Neutral-site / international games
    "Estadio Alfredo Harp Helu":    Park(0, "Estadio Alfredo Harp Helu",     0, 19.4036, -99.0992, 7349, 30,  "open",        1.20, 1.30, 1.10),
    "UNIQLO Field at Dodger Stadium": Park(0, "UNIQLO Field at Dodger Stadium", 119, 34.0739, -118.2400, 510, 25, "open",     1.00, 1.01, 0.99),
}


_LEAGUE_AVG_PARK = Park(0, "league_avg", 0, 39.0, -97.0, 500, 0, "open", 1.00, 1.00, 1.00)


def get_park(venue_name: str) -> Park:
    """Lookup by venue name with graceful fallback for unknown / spring-training parks."""
    if venue_name in PARKS_BY_NAME:
        return PARKS_BY_NAME[venue_name]
    # Substring heuristic — handles minor name changes (sponsor renaming, etc.)
    for k, v in PARKS_BY_NAME.items():
        if k.lower() in venue_name.lower() or venue_name.lower() in k.lower():
            return v
    return _LEAGUE_AVG_PARK
