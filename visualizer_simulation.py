import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="SWEEP Telemetry Explorer", layout="wide")
st.title("🚀 SWEEP Mission Telemetry Explorer (Multi-Run)")

# ==========================================
# 1. DYNAMIC DRILL-DOWN DIRECTORY NAVIGATOR
# ==========================================
RESULTS_DIR = "results"
if not os.path.exists(RESULTS_DIR):
    st.error(f"Directory '{RESULTS_DIR}' not found. Please run a simulation first.")
    st.stop()

st.sidebar.header("1. Navigate Folder Tree")

current_nav_path = RESULTS_DIR
level = 1

# Dynamically generate selectboxes as deep as the folder tree goes
while True:
    try:
        # Find all subdirectories in the current path
        subdirs = [
            d
            for d in os.listdir(current_nav_path)
            if os.path.isdir(os.path.join(current_nav_path, d))
        ]
    except Exception:
        subdirs = []

    if not subdirs:
        break  # Hit the bottom of the tree, no more folders to drill into

    # Provide an option to stop drilling here and look at files
    options = ["📂 [Use This Folder]"] + sorted(subdirs)

    choice = st.sidebar.selectbox(
        f"Level {level} Folder:", options, key=f"nav_lvl_{level}"
    )

    if choice == "📂 [Use This Folder]":
        break  # User locked in their choice

    # Update the path to go one level deeper
    current_nav_path = os.path.join(current_nav_path, choice)
    level += 1

# Show the user exactly where they are
display_loc = os.path.relpath(current_nav_path, RESULTS_DIR)
if display_loc == ".":
    display_loc = "Root (results/)"
st.sidebar.info(f"📍 **Target:** `{display_loc}`")

st.sidebar.header("2. Select Runs")

# Now map all parquet files recursively inside the target folder
path_mapping = {}
for root, _, files in os.walk(current_nav_path):
    for file in files:
        if file.endswith(".parquet"):
            # full_rel is what we use to load the data later
            full_rel = os.path.relpath(os.path.join(root, file), RESULTS_DIR)
            # disp_rel is the short path relative to where we navigated
            disp_rel = os.path.relpath(os.path.join(root, file), current_nav_path)
            path_mapping[full_rel] = disp_rel

file_keys = list(path_mapping.keys())

if not file_keys:
    st.sidebar.warning("No .parquet files found in this folder or its subfolders.")
    st.stop()

# Sort them so they appear in a logical order
file_keys.sort()


def clean_display_name(full_rel_path):
    raw_name = path_mapping[full_rel_path].replace("\\", "/")

    if "/" in raw_name:
        file_name = os.path.basename(raw_name).replace(".parquet", "")
        parent_folder = raw_name.split("/")[-2]

        # If the file is just generically named "telemetry", the folder is the only thing that matters
        if file_name == "telemetry":
            return f"📁 {parent_folder}"
        else:
            return f"📄 {file_name}  📂 [{parent_folder}]"

    return f"📄 {raw_name.replace('.parquet', '')}"


selected_files = st.sidebar.multiselect(
    "Choose runs to compare:",
    file_keys,
    default=[file_keys[0]] if file_keys else [],
    format_func=clean_display_name,
)

if not selected_files:
    st.info("👈 Please select at least one specific simulation run.")
    st.stop()


# ==========================================
# 2. LOAD DATA & UNIT CONVERSION LOGIC
# ==========================================
@st.cache_data
def load_data(file_rel_path):
    parquet_file = os.path.join(RESULTS_DIR, file_rel_path)
    if os.path.exists(parquet_file):
        return pd.read_parquet(parquet_file)
    return None


data_dict = {}
for file_path in selected_files:
    df = load_data(file_path)
    if df is not None:
        data_dict[file_path] = df

if not data_dict:
    st.error("No telemetry data found in the selected files.")
    st.stop()

first_run_key = list(data_dict.keys())[0]
variables = data_dict[first_run_key].columns.tolist()
default_x = "time_days" if "time_days" in variables else variables[0]


def get_unit_options(var_name):
    if var_name.endswith("_rad"):
        return ["Radians", "Degrees"], "rad"
    if var_name.endswith("_m"):
        return ["Meters", "Kilometers"], "m"
    if var_name.endswith("_km"):
        return ["Kilometers", "Meters"], "km"
    if var_name.endswith("_ms"):
        return ["m/s", "km/s"], "ms"
    if var_name.endswith("_sec"):
        return ["Seconds", "Hours", "Days"], "sec"
    return None, None


def apply_conversion(data, unit_choice, suffix):
    if unit_choice == "Degrees":
        return np.rad2deg(data), "[deg]"
    if unit_choice == "Radians":
        return data, "[rad]"
    if unit_choice == "Kilometers" and suffix == "m":
        return data / 1000.0, "[km]"
    if unit_choice == "Meters" and suffix == "m":
        return data, "[m]"
    if unit_choice == "Meters" and suffix == "km":
        return data * 1000.0, "[m]"
    if unit_choice == "Kilometers" and suffix == "km":
        return data, "[km]"
    if unit_choice == "km/s":
        return data / 1000.0, "[km/s]"
    if unit_choice == "m/s":
        return data, "[m/s]"
    if unit_choice == "Hours":
        return data / 3600.0, "[hr]"
    if unit_choice == "Days":
        return data / 86400.0, "[days]"
    if unit_choice == "Seconds":
        return data, "[sec]"
    return data, ""


# ==========================================
# 3. CONFIGURE SUBPLOTS (Dynamic UI)
# ==========================================
st.sidebar.header("2. Build Your Plot")
num_subplots = st.sidebar.number_input(
    "Number of Subplots", min_value=1, max_value=6, value=1
)

plot_configs = []
for i in range(num_subplots):
    st.sidebar.markdown(f"**--- Subplot {i+1} ---**")
    x_var = st.sidebar.selectbox(
        f"X-Axis (Plot {i+1})",
        variables,
        index=variables.index(default_x),
        key=f"x_{i}",
    )
    x_opts, x_suffix = get_unit_options(x_var)
    x_unit = (
        st.sidebar.selectbox(f"↳ {x_var} Unit", x_opts, key=f"x_unit_{i}")
        if x_opts
        else None
    )

    y_vars = st.sidebar.multiselect(
        f"Y-Axis Traces (Plot {i+1})", variables, key=f"y_{i}"
    )

    use_log_y = st.sidebar.checkbox(f"Use Log Scale (Plot {i+1})", key=f"log_y_{i}")

    y_configs = []

    for y in y_vars:
        y_opts, y_suffix = get_unit_options(y)
        unit_choice = (
            st.sidebar.selectbox(f"↳ [{y}] Unit", y_opts, key=f"y_unit_{i}_{y}")
            if y_opts
            else None
        )

        proc_choice = st.sidebar.selectbox(
            f"↳ [{y}] Processing", ["Raw", "Averaged"], key=f"y_proc_{i}_{y}"
        )

        y_configs.append(
            {
                "var": y,
                "unit": unit_choice,
                "suffix": y_suffix,
                "processing": proc_choice,
            }
        )

    plot_configs.append(
        {
            "x": {"var": x_var, "unit": x_unit, "suffix": x_suffix},
            "y": y_configs,
            "log_y": use_log_y,
        }
    )

# --- GLOBAL SETTINGS ---
st.sidebar.header("3. Global Settings")
smoothing_window = st.sidebar.slider(
    "Averaging Window Size",
    min_value=1,
    max_value=500,
    value=100,
    step=10,
    help="Controls the intensity of the rolling mean for variables set to 'Averaged'.",
)

# ==========================================
# 4. RENDER PLOTLY GRAPH
# ==========================================
if any(config["y"] for config in plot_configs):
    fig = make_subplots(
        rows=num_subplots, cols=1, shared_xaxes=True, vertical_spacing=0.05
    )
    color_palette = px.colors.qualitative.Plotly
    line_dash_styles = ["solid", "dash", "dot", "dashdot"]

    for i, config in enumerate(plot_configs):
        row = i + 1
        y_titles = set()

        for run_idx, (file_rel_path, df_run) in enumerate(data_dict.items()):
            run_color = color_palette[run_idx % len(color_palette)]

            short_name = clean_display_name(file_rel_path)

            x_raw = df_run[config["x"]["var"]].copy()
            x_data, x_label = apply_conversion(
                x_raw, config["x"]["unit"], config["x"]["suffix"]
            )
            x_title = f"{config['x']['var']} {x_label}"

            for j, y_conf in enumerate(config["y"]):
                y_raw = df_run[y_conf["var"]].copy()

                is_averaged = y_conf["processing"] == "Averaged"
                if is_averaged and smoothing_window > 1:
                    y_raw = y_raw.rolling(
                        window=smoothing_window, min_periods=1, center=True
                    ).mean()

                y_data, y_label = apply_conversion(
                    y_raw, y_conf["unit"], y_conf["suffix"]
                )
                line_style = line_dash_styles[j % len(line_dash_styles)]

                trace_name = f"[{short_name}] {y_conf['var']}" + (
                    " (Avg)" if is_averaged else ""
                )

                fig.add_trace(
                    go.Scatter(
                        x=x_data,
                        y=y_data,
                        mode="lines",
                        name=trace_name,
                        line=dict(color=run_color, dash=line_style),
                        legendgroup=file_rel_path,
                    ),
                    row=row,
                    col=1,
                )
                y_titles.add(f"{y_conf['var']} {y_label}")

        fig.update_xaxes(title_text=x_title, row=row, col=1)

        fig.update_yaxes(
            title_text=list(y_titles)[0] if len(y_titles) == 1 else "Multiple Values",
            type="log" if config["log_y"] else "linear",
            row=row,
            col=1,
        )

    fig.update_layout(
        height=max(500, 350 * num_subplots),
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=80),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            groupclick="toggleitem",
        ),
    )
    fig.update_yaxes(exponentformat="power", showexponent="all")
    st.plotly_chart(fig, width="stretch")
else:
    st.info(
        "👈 Select at least one Y-Axis variable in the sidebar to generate the plot."
    )
