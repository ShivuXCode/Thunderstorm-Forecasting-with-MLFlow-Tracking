FEATURE_COLUMNS = [
    "SWEAT index",
    "K index",
    "Totals totals index",
    "Environmental_Stability",
    "Moisture_Indices",
    "Convective_Potential",
    "Temperature_Pressure",
    "Moisture_Temperature_Profiles",
]


SAMPLE_PRESETS = {
    "Calm baseline": {
        "SWEAT index": 161.9,
        "K index": 24.0,
        "Totals totals index": 40.6,
        "Environmental_Stability": 6.6,
        "Moisture_Indices": 33.9,
        "Convective_Potential": 0.0,
        "Temperature_Pressure": 5744.0,
        "Moisture_Temperature_Profiles": 950.8,
    },
    "Storm-prone sample": {
        "SWEAT index": 238.5,
        "K index": 35.25,
        "Totals totals index": 44.2,
        "Environmental_Stability": 2.0,
        "Moisture_Indices": 54.2,
        "Convective_Potential": 1044.65,
        "Temperature_Pressure": 5796.0,
        "Moisture_Temperature_Profiles": 960.8,
    },
    "High-risk sample": {
        "SWEAT index": 307.37,
        "K index": 41.7,
        "Totals totals index": 51.4,
        "Environmental_Stability": -6.4,
        "Moisture_Indices": 69.3,
        "Convective_Potential": 3288.65,
        "Temperature_Pressure": 5845.0,
        "Moisture_Temperature_Profiles": 991.25,
    },
}


DEFAULT_PRESET = "Calm baseline"
