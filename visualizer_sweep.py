import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Phase Sweep Explorer", layout="wide")
st.title("📊 Q-Law Phase Convergence Sweep Explorer")
st.markdown(
    "Compare multiple phase sweeps (e.g., Best, Medium, Worst lighting) and analyze convergence trends."
)

# ==========================================
# 1. DYNAMIC DRILL-DOWN DIRECTORY NAVIGATOR
# ==========================================
RESULTS_DIR = "results"
if not os.path.exists(RESULTS_DIR):
    st.error(f"Directory '{RESULTS_DIR}' not found. Please run a sweep first.")
    st.stop()

st.sidebar.header("1. Navigate Folder Tree")

current_nav_path = RESULTS_DIR
level = 1

while True:
    try:
        subdirs = [
            d
            for d in os.listdir(current_nav_path)
            if os.path.isdir(os.path.join(current_nav_path, d))
        ]
    except Exception:
        subdirs = []

    if not subdirs:
        break

    options = ["📂 [Use This Folder]"] + sorted(subdirs)

    choice = st.sidebar.selectbox(
        f"Level {level} Folder:", options, key=f"nav_lvl_{level}"
    )

    if choice == "📂 [Use This Folder]":
        break

    current_nav_path = os.path.join(current_nav_path, choice)
    level += 1

display_loc = os.path.relpath(current_nav_path, RESULTS_DIR)
if display_loc == ".":
    display_loc = "Root (results/)"
st.sidebar.info(f"📍 **Target:** `{display_loc}`")

st.sidebar.header("2. Select Sweeps")

path_mapping = {}
for root, _, files in os.walk(current_nav_path):
    for file in files:
        if file.endswith(".csv"):
            full_rel = os.path.relpath(os.path.join(root, file), RESULTS_DIR)
            disp_rel = os.path.relpath(os.path.join(root, file), current_nav_path)
            path_mapping[full_rel] = disp_rel

file_keys = list(path_mapping.keys())

if not file_keys:
    st.sidebar.warning("No .csv sweep files found in this folder or its subfolders.")
    st.stop()

file_keys.sort(
    key=lambda x: os.path.getmtime(os.path.join(RESULTS_DIR, x)), reverse=True
)


st.sidebar.markdown("---")
st.sidebar.markdown("**B. Choose runs to overlay:**")

from collections import defaultdict

files_by_folder = defaultdict(list)

for path in file_keys:
    raw_name = path_mapping[path].replace("\\", "/")
    if "/" in raw_name:
        folder = os.path.dirname(raw_name)
        filename = os.path.basename(raw_name)
    else:
        folder = "/"
        filename = raw_name
    files_by_folder[folder].append((path, filename))

selected_files = []
is_first_folder = True

for folder, files in sorted(files_by_folder.items()):
    with st.sidebar.expander(f"📁 {folder}", expanded=is_first_folder):
        for full_path, filename in sorted(files):
            default_check = is_first_folder and files.index((full_path, filename)) == 0

            if st.checkbox(f"📄 {filename}", value=default_check, key=full_path):
                selected_files.append(full_path)

    is_first_folder = False

if not selected_files:
    st.info("👈 Please select at least one CSV file from the sidebar.")
    st.stop()

# ==========================================
# 2. UI CONTROLS (Zooming & Processing)
# ==========================================
st.sidebar.header("3. View Settings")

col1, col2 = st.sidebar.columns(2)
with col1:
    x_min = st.number_input("Min Phase (deg)", value=-180, step=10)
with col2:
    x_max = st.number_input("Max Phase (deg)", value=180, step=10)

# NEW: Data Normalization Section
st.sidebar.header("4. Data Normalization")
plot_relative = st.sidebar.checkbox(
    "Plot Relative ToF", value=False, help="Normalize Time of Flight by Stage 1 ToF"
)
stage_1_tof = 1.0

if plot_relative:
    stage_1_tof = st.sidebar.number_input(
        "Stage 1 ToF (Days)",
        value=20.0,
        step=1.0,
        help="The duration of Stage 1. Stage 2 absolute ToF will be divided by this value.",
    )
    # Prevent division by zero
    if stage_1_tof <= 0:
        st.sidebar.error("Stage 1 ToF must be strictly positive.")
        stage_1_tof = 1.0

st.sidebar.header("5. Data Processing")
show_raw = st.sidebar.checkbox("Show Raw Data Points", value=True)

smoothing_window = st.sidebar.slider(
    "Averaging Window Size",
    min_value=1,
    max_value=20,
    value=1,
    step=1,
    help="1 = No smoothing.",
)

st.sidebar.header("6. Curve Fitting")
show_poly = st.sidebar.checkbox("Overlay Polynomial Fit", value=False)
if show_poly:
    poly_degree = st.sidebar.number_input(
        "Polynomial Degree",
        min_value=1,
        max_value=15,
        value=4,
        step=1,
        help="Higher degrees bend more to fit the data. 2=Parabola, 3=Cubic, etc.",
    )

st.sidebar.markdown("---")
timeout_limit = st.sidebar.number_input(
    "Absolute Timeout Limit (Days)", value=150.0, step=10.0
)

# ==========================================
# 3. RENDER PLOTLY GRAPH
# ==========================================
fig = go.Figure()
color_palette = px.colors.qualitative.Plotly

for idx, file_rel_path in enumerate(selected_files):
    file_path = os.path.join(RESULTS_DIR, file_rel_path)
    df = pd.read_csv(file_path)

    # --- DYNAMIC METRIC SELECTOR ---
    available_metrics = [
        col for col in df.columns if "ToF" in col or "Time_of_Flight" in col
    ]

    if "True_Delta_u_deg" not in df.columns or not available_metrics:
        st.error(f"File {file_rel_path} is missing required Time of Flight columns.")
        continue

    if "tof_metric" not in st.session_state:
        st.session_state.tof_metric = available_metrics[0]

    if idx == 0:
        st.sidebar.markdown("---")
        metric_choice = st.sidebar.selectbox(
            "Select ToF Metric to Plot", available_metrics
        )
        st.session_state.tof_metric = metric_choice
    # -------------------------------

    df = df.sort_values(by="True_Delta_u_deg")
    df_filtered = df[
        (df["True_Delta_u_deg"] >= x_min) & (df["True_Delta_u_deg"] <= x_max)
    ].copy()

    x_data = df_filtered["True_Delta_u_deg"].values

    # Use the dynamically selected metric
    y_raw = df_filtered[st.session_state.tof_metric].values

    # Apply relative ToF scaling
    if plot_relative:
        y_raw = y_raw / stage_1_tof

    run_color = color_palette[idx % len(color_palette)]

    raw_name = path_mapping[file_rel_path].replace("\\", "/")
    if "/" in raw_name:
        folder = os.path.basename(os.path.dirname(raw_name))
        file_name = os.path.basename(raw_name).replace(".csv", "")
        short_name = f"[{folder}] {file_name}"
    else:
        short_name = raw_name.replace(".csv", "")

    if show_raw:
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_raw,
                mode="lines+markers",
                name=f"{short_name} (Raw)",
                line=dict(color=run_color, width=1.5),
                marker=dict(size=5, opacity=0.8),
                legendgroup=file_rel_path,
            )
        )

    if smoothing_window > 1:
        y_smoothed = (
            pd.Series(y_raw)
            .rolling(window=smoothing_window, center=True, min_periods=1)
            .mean()
        )
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_smoothed,
                mode="lines",
                name=f"{short_name} (Smoothed Avg)",
                line=dict(color=run_color, width=3),
                legendgroup=file_rel_path,
            )
        )

    if show_poly and len(x_data) > poly_degree:
        valid_mask = ~np.isnan(x_data) & ~np.isnan(y_raw)
        x_clean = x_data[valid_mask]
        y_clean = y_raw[valid_mask]

        if len(x_clean) > poly_degree:
            coefs = np.polyfit(x_clean, y_clean, poly_degree)
            poly_func = np.poly1d(coefs)

            x_curve = np.linspace(x_clean.min(), x_clean.max(), 300)
            y_curve = poly_func(x_curve)

            fig.add_trace(
                go.Scatter(
                    x=x_curve,
                    y=y_curve,
                    mode="lines",
                    name=f"{short_name} (Poly d={poly_degree})",
                    line=dict(color=run_color, width=2.5, dash="dash"),
                    legendgroup=file_rel_path,
                )
            )

# Scale the visual timeout limit if relative plotting is active
display_timeout = timeout_limit / stage_1_tof if plot_relative else timeout_limit

fig.add_hline(
    y=display_timeout,
    line_dash="dot",
    line_color="red",
    annotation_text="Timeout Boundary",
    annotation_position="bottom right",
)

# Dynamically change the Y-axis label to match the selected metric
y_axis_label = (
    f"Relative {st.session_state.tof_metric} (T / T_stage1) [-]"
    if plot_relative
    else f"{st.session_state.tof_metric} [Days]"
)

fig.update_layout(
    height=700,
    template="plotly_white",
    xaxis_title="Initial Phase Offset Δu [deg]",
    yaxis_title=y_axis_label,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=40, r=40, t=60, b=40),
)

st.plotly_chart(fig, width="stretch")
