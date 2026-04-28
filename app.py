import streamlit as st  # frontend UI

from app.predictor import predict_weather
from app.sample_inputs import DEFAULT_PRESET, FEATURE_COLUMNS, SAMPLE_PRESETS


FEATURE_RANGES = {
    "SWEAT index": (0.0, 450.0),
    "K index": (-20.0, 60.0),
    "Totals totals index": (0.0, 70.0),
    "Environmental_Stability": (-30.0, 40.0),
    "Moisture_Indices": (0.0, 100.0),
    "Convective_Potential": (-1000.0, 6000.0),
    "Temperature_Pressure": (5200.0, 6100.0),
    "Moisture_Temperature_Profiles": (850.0, 1050.0),
}

FEATURE_HELP = {
    "SWEAT index": "Severe Weather Threat Index: Combines low-level wind shear, mid-level instability, and moisture to assess thunderstorm severity potential. Higher values (200+) indicate strong thunderstorm threat.",
    "K index": "Thunderstorm Potential Index: Measures the difference between upper and lower atmospheric temperatures plus moisture at 850mb. Higher values (30+) suggest better conditions for thunderstorm development.",
    "Totals totals index": "Convective Instability Index: Calculates the temperature difference between upper-level air and surface air. Values above 40 strongly favor thunderstorm development and severity.",
    "Environmental_Stability": "Atmospheric Stability: Represents how resistant the atmosphere is to vertical motion. Negative values (<-10) indicate unstable conditions favorable for storm development; positive values suggest stable air.",
    "Moisture_Indices": "Atmospheric Moisture Level: Indicates the amount of water vapor available in the lower atmosphere. Higher values (>60) provide more fuel for storm development and precipitation.",
    "Convective_Potential": "Buoyant Energy Availability: Estimates the energy available to lift air parcels vertically. Higher positive values (>2000) indicate stronger updrafts and more intense storm potential.",
    "Temperature_Pressure": "Temperature-Pressure Profile: Captures the vertical distribution of temperature across pressure levels. Specific ranges indicate favorable lapse rates for convection and storm organization.",
    "Moisture_Temperature_Profiles": "Vertical Moisture-Temperature Structure: Represents how moisture and temperature vary from the surface to upper atmosphere. Specific values indicate the stability and structure of the atmospheric column for storm formation.",
}


def _initialize_input_state() -> None:
    default_values = SAMPLE_PRESETS[DEFAULT_PRESET]
    for feature_name in FEATURE_COLUMNS:
        if feature_name not in st.session_state:
            st.session_state[feature_name] = float(default_values[feature_name])


def _apply_preset(preset_name: str) -> None:
    for feature_name, value in SAMPLE_PRESETS[preset_name].items():
        st.session_state[feature_name] = float(value)


def _reset_inputs_to_zero() -> None:
    for feature_name in FEATURE_COLUMNS:
        st.session_state[feature_name] = 0.0


def _initialize_prediction_state() -> None:
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None


st.set_page_config(page_title="Thunderstorm Predictor", page_icon="⛈️", layout="wide")
_initialize_input_state()
_initialize_prediction_state()

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        .hero {
            padding: 1.4rem 1.5rem;
            border-radius: 20px;
            background: linear-gradient(135deg, rgba(20,24,50,0.98), rgba(32,78,122,0.95));
            color: white;
            border: 1px solid rgba(255,255,255,0.12);
            margin-bottom: 1rem;
        }
        .hero h1 { margin: 0; font-size: 2.1rem; }
        .hero p { margin: 0.35rem 0 0; opacity: 0.86; }
        .card {
            background: white;
            border-radius: 18px;
            padding: 1rem 1.1rem;
            border: 1px solid #e7eaf0;
            box-shadow: 0 10px 30px rgba(20, 24, 50, 0.08);
        }
        section[data-testid="stSidebar"] { background: #0f172a; }
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] span {
            color: #f8fafc !important;
        }
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
            background: #1e293b !important;
            color: #f8fafc !important;
            border-color: #334155 !important;
        }
        section[data-testid="stSidebar"] div[data-baseweb="select"] input {
            color: #f8fafc !important;
        }
        section[data-testid="stSidebar"] button {
            border-color: #334155 !important;
        }
        section[data-testid="stSidebar"] div.stButton > button {
            background: #1e293b !important;
            color: #f8fafc !important;
            border: 1px solid #475569 !important;
        }
        section[data-testid="stSidebar"] div.stButton > button p {
            color: #f8fafc !important;
        }
        section[data-testid="stSidebar"] div.stButton > button:hover {
            background: #334155 !important;
            color: #f8fafc !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>Thunderstorm Predictor</h1>
      <p>Load a realistic preset or fine-tune the inputs to test the saved KNN model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Quick presets")
    selected_preset = st.selectbox("Preset", list(SAMPLE_PRESETS.keys()), index=list(SAMPLE_PRESETS.keys()).index(DEFAULT_PRESET))

    action_left, action_right = st.columns(2)
    with action_left:
        load_clicked = st.button("Load preset", use_container_width=True)
    with action_right:
        reset_clicked = st.button("Reset", use_container_width=True)

    if load_clicked:
        _apply_preset(selected_preset)
        st.rerun()

    if reset_clicked:
        _reset_inputs_to_zero()
        st.rerun()

    st.caption("These values come from the processed dataset medians and high-risk quantiles.")

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Input features")
    st.write("Edit the values below, then run a prediction.")

    with st.form("prediction_form"):
        inputs = {}
        columns = st.columns(2)
        for index, column in enumerate(FEATURE_COLUMNS):
            with columns[index % 2]:
                min_value, max_value = FEATURE_RANGES[column]
                # Explicitly use the session state value as the default
                current_value = st.session_state.get(column, float(SAMPLE_PRESETS[DEFAULT_PRESET][column]))
                inputs[column] = st.number_input(
                    f"{column} ({min_value:.0f} to {max_value:.0f})",
                    key=column,
                    value=current_value,
                    help=FEATURE_HELP[column],
                    format="%.2f",
                )
        submitted = st.form_submit_button("Predict thunderstorm risk", use_container_width=True)
        
        if submitted:
            st.session_state.prediction_result = predict_weather(inputs)

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Result")
    if st.session_state.prediction_result is not None:
        result = st.session_state.prediction_result
        label = "Thunderstorm likely" if result["prediction"] == 1 else "No thunderstorm likely"
        st.metric("Prediction", label)
        st.metric("Probability", f"{result['probability']:.4f}")
        st.caption(f"Model source: {result['model_path']}")
        
        # Debug: Show the actual input values used
        with st.expander("📊 Debug: Input values used"):
            debug_inputs = {}
            for feature in FEATURE_COLUMNS:
                debug_inputs[feature] = st.session_state.get(feature, "N/A")
            st.json(debug_inputs)
    else:
        st.info("Choose a preset on the left or keep the defaults, then run a prediction.")
    st.markdown('</div>', unsafe_allow_html=True)