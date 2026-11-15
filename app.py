import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import pickle

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="VOLTEX | Battery Diagnostics",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;600;700;800&display=swap');

.stApp {
    background: #090e19 !important;
    color: #f4f7ff !important;
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    padding-top: 1.5rem;
    max-width: 1500px;
}

/* Headings */
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4 {
    color: #ffffff !important;
    font-family: 'Manrope', sans-serif;
}

/* Normal text */
.stApp p,
.stApp label,
.stApp li {
    color: #e1e9f5 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #101827 !important;
    border-right: 1px solid #34445e;
}

[data-testid="stSidebar"] * {
    color: #f1f5fc !important;
}

/* Captions */
[data-testid="stCaptionContainer"] p {
    color: #b9c8dc !important;
}

/* Metric labels and values */
[data-testid="stMetricLabel"] {
    color: #b9c8dc !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

/* Panels */
.panel {
    background: linear-gradient(145deg, #1c2940, #141e30);
    border: 1px solid #34445e;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 15px;
}

.panel-title {
    color: #c0cee2 !important;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
}

/* Vehicle section */
.vehicle-stage {
    background: radial-gradient(
        ellipse at center,
        #263b5c,
        #17253b 55%,
        #111a2a
    );
    border: 1px solid #34445e;
    border-radius: 17px;
    padding: 25px;
    min-height: 270px;
    text-align: center;
}

/* Buttons */
.stButton button {
    background: #438cf0 !important;
    color: white !important;
    border-radius: 9px;
}

/* Dropdown */
div[data-baseweb="select"] {
    background: #18243a !important;
}

div[data-baseweb="select"] * {
    color: white !important;
}

/* Data table */
[data-testid="stDataFrame"] {
    color: white !important;
}

/* Alerts */
[data-testid="stAlert"] p {
    color: white !important;
}

hr {
    border-color: #34445e !important;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# DATASET LOADING
# =====================================================

@st.cache_data
def load_data():

    possible_files = [
        BASE_DIR / "data" / "processed" / "soh_model_dataset.csv",
        BASE_DIR / "data" / "processed" / "battery_features.csv",
        BASE_DIR / "data" / "raw" / "all_batteries_capacity_soh.csv"
    ]

    for file_path in possible_files:

        if file_path.exists():

            try:
                data = pd.read_csv(file_path)
                return data, file_path.name

            except Exception as e:
                st.error(f"Error reading dataset: {e}")

    return pd.DataFrame(), None


df, source_file = load_data()

if df.empty:

    st.error(
        "Dataset not found. Please check your CSV file location."
    )

    st.code("""
supervisedproject/
│
├── app.py
│
└── data/
    ├── processed/
    │   ├── soh_model_dataset.csv
    │   └── battery_features.csv
    │
    └── raw/
        └── all_batteries_capacity_soh.csv
""")

    st.stop()


# =====================================================
# NORMALIZE COLUMN NAMES
# =====================================================

df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
    .str.replace("-", "_", regex=False)
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
    .str.replace("%", "", regex=False)
)

# Handle common column-name variations

column_mapping = {

    "battery": "battery_id",
    "battery_name": "battery_id",
    "cell_id": "battery_id",

    "cycle": "discharge_cycle",
    "cycle_number": "discharge_cycle",
    "discharge_cycles": "discharge_cycle",

    "soh_percent": "soh",
    "soh_percentage": "soh",
    "state_of_health": "soh",
    "state_of_health_percent": "soh",

    "capacity": "capacity_ah",
    "discharge_capacity": "capacity_ah",
    "discharge_capacity_ah": "capacity_ah"
}

df = df.rename(columns=column_mapping)
# =====================================================
# MERGE CAPACITY DATA FROM ORIGINAL DATASET
# =====================================================

capacity_file = (
    BASE_DIR / "data" / "raw"
    / "all_batteries_capacity_soh.csv"
)

if capacity_file.exists():

    capacity_df = pd.read_csv(capacity_file)

    # Normalize original dataset columns
    capacity_df.columns = (
        capacity_df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("%", "", regex=False)
    )

    # Rename capacity dataset columns
    capacity_mapping = {
        "battery": "battery_id",
        "battery_name": "battery_id",
        "cycle": "discharge_cycle",
        "cycle_number": "discharge_cycle",
        "capacity": "capacity_ah",
        "discharge_capacity": "capacity_ah",
        "discharge_capacity_ah": "capacity_ah"
    }

    capacity_df = capacity_df.rename(
        columns=capacity_mapping
    )

    required = [
        "battery_id",
        "discharge_cycle",
        "capacity_ah"
    ]

    if all(col in capacity_df.columns for col in required):

        capacity_df["battery_id"] = (
            capacity_df["battery_id"].astype(str).str.strip()
        )

        df["battery_id"] = (
            df["battery_id"].astype(str).str.strip()
        )

        capacity_df["discharge_cycle"] = pd.to_numeric(
            capacity_df["discharge_cycle"],
            errors="coerce"
        )

        capacity_df["capacity_ah"] = pd.to_numeric(
            capacity_df["capacity_ah"],
            errors="coerce"
        )

        capacity_df = capacity_df.dropna(
            subset=required
        )

        capacity_df = capacity_df.drop_duplicates(
            subset=["battery_id", "discharge_cycle"]
        )

        # Merge only if capacity is missing
        if "capacity_ah" not in df.columns:

            df = df.merge(
                capacity_df[required],
                on=["battery_id", "discharge_cycle"],
                how="left"
            )

            st.success("Original capacity data merged successfully.")

        else:
            st.info("Capacity column already exists.")

    else:
        st.warning(
            "Required columns not found in original capacity dataset."
        )


# =====================================================
# IDENTIFY REQUIRED COLUMNS
# =====================================================

# Battery ID

if "battery_id" not in df.columns:

    st.error("Battery ID column not found.")

    st.write("Available columns:", df.columns.tolist())

    st.stop()


# Discharge cycle

if "discharge_cycle" not in df.columns:

    st.error("Discharge cycle column not found.")

    st.write("Available columns:", df.columns.tolist())

    st.stop()


# Convert numeric columns

df["discharge_cycle"] = pd.to_numeric(
    df["discharge_cycle"],
    errors="coerce"
)


# =====================================================
# CALCULATE SOH IF REQUIRED
# =====================================================

if "soh" not in df.columns:

    if "capacity_ah" in df.columns:

        df["capacity_ah"] = pd.to_numeric(
            df["capacity_ah"],
            errors="coerce"
        )

        df = df.sort_values(
            ["battery_id", "discharge_cycle"]
        )

        df["soh"] = df.groupby("battery_id")[
            "capacity_ah"
        ].transform(
            lambda x: (
                x / x.dropna().iloc[0] * 100
                if x.notna().any()
                and x.dropna().iloc[0] != 0
                else np.nan
            )
        )

    else:

        st.error(
            "SOH and capacity columns are both missing."
        )

        st.write(
            "Available columns:",
            df.columns.tolist()
        )

        st.stop()


df["soh"] = pd.to_numeric(
    df["soh"],
    errors="coerce"
)

if "capacity_ah" in df.columns:

    df["capacity_ah"] = pd.to_numeric(
        df["capacity_ah"],
        errors="coerce"
    )

df = df.dropna(
    subset=[
        "battery_id",
        "discharge_cycle",
        "soh"
    ]
)

df = df.sort_values(
    ["battery_id", "discharge_cycle"]
)

if df.empty:
    st.error("No valid battery records found.")
    st.stop()
# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown(
        """
        <h2 style='color:#69aaff'>
        ⬡ VOLTEX
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.caption("BATTERY INTELLIGENCE SYSTEM")

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "Diagnostic",
            "Battery Explorer",
            "Model Performance",
            "Data Laboratory"
        ]
    )

    st.markdown("---")

    st.markdown("### DATA SOURCE")

    st.markdown("NASA Lithium-Ion Dataset")

    st.caption("Historical laboratory measurements")

    st.success("LAB MODE · OFFLINE DATA")


# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div style="
display:flex;
justify-content:space-between;
align-items:center;
border-bottom:1px solid #29364b;
padding-bottom:15px;
margin-bottom:25px;
">

<div>
<span style="color:#65a7ff;font-size:25px">⬡</span>

<span style="font-size:22px;font-weight:800;
letter-spacing:3px">VOLTEX</span>

<span style="color:#8395b0;font-size:11px;
margin-left:15px">BATTERY DIAGNOSTICS</span>
</div>

<div>
<span style="background:#18334d;color:#8ec5ff;
padding:7px 12px;border-radius:20px;font-size:11px">
NASA DATASET
</span>
</div>

</div>
""", unsafe_allow_html=True)


# =====================================================
# BATTERY SELECTION
# =====================================================

batteries = sorted(
    df["battery_id"].astype(str).unique()
)

selected_battery = st.sidebar.selectbox(
    "SELECT BATTERY",
    batteries
)

battery_df = df[
    df["battery_id"].astype(str) == selected_battery
].sort_values("discharge_cycle")

latest = battery_df.iloc[-1]

first = battery_df.iloc[0]

latest_soh = float(latest["soh"])

latest_cycle = int(latest["discharge_cycle"])

health_status = (
    "HEALTHY" if latest_soh >= 85
    else "DEGRADING" if latest_soh >= 70
    else "BELOW 70% THRESHOLD"
)
# =====================================================
# DIAGNOSTIC PAGE
# =====================================================

if page == "Diagnostic":

    st.caption("VEHICLE SYSTEM / BATTERY HEALTH")

    st.title("Diagnostic")

    st.markdown(
        f"<span class='status'>{health_status}</span>",
        unsafe_allow_html=True
    )

    st.write("")

    left, right = st.columns(
        [1.7, 1],
        gap="large"
    )

    # ----------------------------------
    # LEFT SECTION
    # ----------------------------------

    with left:

        st.markdown("""
        <div class="vehicle-stage">

        <div class="panel-title">
        BATTERY SYSTEM VISUALIZATION
        </div>

        <div style="font-size:105px;margin-top:20px">
        🚘
        </div>

        <div style="
        color:#91c6ff;
        font-size:13px;
        margin-top:10px;
        ">
        ELECTRIC VEHICLE BATTERY SYSTEM
        </div>

        <div style="
        color:#8296b3;
        font-size:10px;
        margin-top:8px;
        ">
        HISTORICAL LAB DATA · NOT LIVE TELEMETRY
        </div>

        </div>
        """, unsafe_allow_html=True)

        st.write("")

        c1, c2 = st.columns(2)

        # SOH CHART

        with c1:

            st.markdown(
                """
                <div class='panel'>
                <div class='panel-title'>
                SOH DEGRADATION
                </div>
                """,
                unsafe_allow_html=True
            )

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=battery_df["discharge_cycle"],
                    y=battery_df["soh"],
                    mode="lines",
                    line=dict(
                        color="#65aaff",
                        width=3
                    ),
                    fill="tozeroy",
                    fillcolor="rgba(101,170,255,0.10)"
                )
            )

            fig.add_hline(
                y=70,
                line_dash="dash",
                line_color="#f5b76d"
            )

            fig.update_layout(
                height=260,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#aab9ce",
                margin=dict(l=5, r=5, t=15, b=5),
                xaxis_title="Cycle",
                yaxis_title="SOH (%)"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.markdown("</div>", unsafe_allow_html=True)


        # CAPACITY CHART

        with c2:

            st.markdown(
                """
                <div class='panel'>
                <div class='panel-title'>
                CAPACITY TREND
                </div>
                """,
                unsafe_allow_html=True
            )

            if (
                "capacity_ah" in battery_df.columns
                and battery_df["capacity_ah"].notna().any()
            ):

                fig2 = px.line(
                    battery_df,
                    x="discharge_cycle",
                    y="capacity_ah"
                )

                fig2.update_traces(
                    line_color="#58d6e8",
                    line_width=3
                )

                fig2.update_layout(
                    height=260,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#aab9ce",
                    margin=dict(l=5, r=5, t=15, b=5)
                )

                st.plotly_chart(
                    fig2,
                    use_container_width=True
                )

            else:

                st.info("Capacity data unavailable.")

            st.markdown("</div>", unsafe_allow_html=True)


    # ----------------------------------
    # RIGHT SECTION
    # ----------------------------------

    with right:

        st.markdown(
            """
            <div class='panel'>
            <div class='panel-title'>
            BATTERY CONDITION
            </div>
            """,
            unsafe_allow_html=True
        )

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=latest_soh,

                number={
                    "suffix": "%",
                    "font": {
                        "size": 40,
                        "color": "white"
                    }
                },

                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickcolor": "#94a5bf"
                    },

                    "bar": {
                        "color": (
                            "#65d6a2"
                            if latest_soh >= 70
                            else "#ff777f"
                        )
                    },

                    "steps": [
                        {
                            "range": [0, 70],
                            "color": "#3c2734"
                        },
                        {
                            "range": [70, 85],
                            "color": "#473b2d"
                        },
                        {
                            "range": [85, 100],
                            "color": "#1e403a"
                        }
                    ]
                }
            )
        )

        gauge.update_layout(
            height=220,
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            margin=dict(l=10, r=10, t=20, b=5)
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

        st.markdown("---")

        if (
            "capacity_ah" in battery_df.columns
            and battery_df["capacity_ah"].notna().any()
        ):

            initial_capacity = first["capacity_ah"]

            current_capacity = latest["capacity_ah"]

            m1, m2 = st.columns(2)

            m1.metric(
                "Initial Capacity",
                f"{initial_capacity:.3f} Ah"
            )

            m2.metric(
                "Current Capacity",
                f"{current_capacity:.3f} Ah"
            )

        st.markdown("---")

        st.markdown("### Diagnostic Information")

        st.write(f"**Battery:** {selected_battery}")

        st.write(f"**Recorded Cycles:** {len(battery_df)}")

        st.write(f"**Latest Cycle:** {latest_cycle}")

        st.write(
            f"**SOH Reduction:** {100-latest_soh:.2f}%"
        )

        if latest_soh < 70:

            st.error(
                "Battery is below the project's "
                "70% reference threshold."
            )

        elif latest_soh < 85:

            st.warning(
                "Battery shows noticeable degradation."
            )

        else:

            st.success(
                "Battery is above the selected "
                "reference bands."
            )

        st.caption(
            "The 70% threshold is a project reference, "
            "not a universal battery safety limit."
        )

        st.markdown("</div>", unsafe_allow_html=True)
# =====================================================
# BATTERY EXPLORER
# =====================================================

elif page == "Battery Explorer":

    st.title("Battery Explorer")

    st.caption(
        "Comparative analysis of historical battery degradation."
    )

    fig = px.line(
        df,
        x="discharge_cycle",
        y="soh",
        color="battery_id",
        title="SOH Comparison Across Batteries"
    )

    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color="#f5b76d"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#aab9ce",
        height=420
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    summary = df.groupby("battery_id").agg(
        total_cycles=("discharge_cycle", "count"),
        initial_soh=("soh", "first"),
        final_soh=("soh", "last"),
        minimum_soh=("soh", "min")
    ).reset_index()

    summary["SOH_Drop"] = (
        summary["initial_soh"] -
        summary["final_soh"]
    )

    st.subheader("Battery Comparison")

    st.dataframe(
        summary.round(2),
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# MODEL PERFORMANCE
# =====================================================

elif page == "Model Performance":

    st.title("Model Performance")

    st.caption(
        "Leave-One-Battery-Out cross-validation results."
    )

    metrics = pd.DataFrame({

        "Model": [
            "Extra Trees",
            "Ridge Regression",
            "Random Forest"
        ],

        "MAE": [
            2.6213,
            3.2859,
            3.4927
        ],

        "RMSE": [
            3.8481,
            3.8362,
            4.7298
        ],

        "R2": [
            0.8759,
            0.8766,
            0.8125
        ]

    })

    st.dataframe(
        metrics,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Extra Trees has the lowest MAE. "
        "Ridge has slightly lower RMSE and slightly higher R²."
    )

    st.caption(
        "These are exploratory results from four batteries. "
        "R² is not prediction accuracy."
    )

    st.subheader("Important Features")

    importance = pd.DataFrame({

        "Feature": [
            "voltage_mean",
            "discharge_duration",
            "discharge_cycle",
            "voltage_load_mean",
            "temperature_std"
        ],

        "Importance": [
            2.3748,
            0.9952,
            0.5594,
            0.3904,
            0.3075
        ]

    })

    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Permutation Feature Importance"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#aab9ce"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
# =====================================================
# DATA LABORATORY
# =====================================================

elif page == "Data Laboratory":

    st.title("Data Laboratory")

    st.caption(
        "Inspect the dataset used by the battery health model."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Records",
        len(df)
    )

    c2.metric(
        "Battery Units",
        df["battery_id"].nunique()
    )

    c3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(100),
        use_container_width=True
    )

    st.subheader("Dataset Information")

    info = pd.DataFrame({

        "Column": df.columns,

        "Data Type": [
            str(x) for x in df.dtypes
        ],

        "Missing Values": [
            int(df[col].isnull().sum())
            for col in df.columns
        ]

    })

    st.dataframe(
        info,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Dataset",
        df.to_csv(index=False),
        "battery_dataset.csv",
        "text/csv"
    )


# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<style>


/* DEFAULT DARK BACKGROUND TEXT - WHITE */
.stApp,
.stApp p,
.stApp span,
.stApp label,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp li {
    color: #F1F5FC !important;
    -webkit-text-fill-color: #F1F5FC !important;
}

/* WHITE BACKGROUND PANELS - BLACK TEXT */
.stApp [style*="background: white"],
.stApp [style*="background:#fff"],
.stApp [style*="background: #fff"],
.stApp [style*="background-color: white"],
.stApp [style*="background-color: #ffffff"],
.stApp [style*="background-color:#ffffff"] {
    color: #000000 !important;
}

/* DATA TABLE */
[data-testid="stDataFrame"] {
    color: #000000 !important;
}

/* WHITE BACKGROUND INPUTS */
.stApp input,
.stApp textarea {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

/* SIDEBAR - KEEP WHITE TEXT */
[data-testid="stSidebar"] * {
    color: #F1F5FC !important;
    -webkit-text-fill-color: #F1F5FC !important;
}

/* BUTTONS */
.stButton button {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* Streamlit alerts use light backgrounds, so keep their text dark. */
[data-testid="stAlert"] p,
[data-testid="stAlert"] span {
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
}

</style>
""", unsafe_allow_html=True)

