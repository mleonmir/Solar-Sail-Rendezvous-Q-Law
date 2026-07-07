import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tudatpy.astro import element_conversion

from config import DEFAULT_MU_EARTH


# ==========================================
# MAIN PLOTTING FUNCTION
# ==========================================
def plot_mission(state_array, dep_array, sim_config):
    print("\nGenerating interactive Plotly mission plots (Universal Telemetry)...")
    log_cfg = sim_config.logging

    time_sec = dep_array[:, 0] - dep_array[0, 0]
    time_days = time_sec / 86400.0

    # Spacecraft Cartesian State
    x_km = state_array[:, 1] / 1000.0
    y_km = state_array[:, 2] / 1000.0
    z_km = state_array[:, 3] / 1000.0

    # ==========================================
    # STATIC DATA EXTRACTION (Universal Array)
    # ==========================================
    # Keplerian State (Cols 1-6) - Kept in Radians/Meters for precise Mean Calc
    c_a_m = dep_array[:, 1]
    c_e = dep_array[:, 2]
    c_i_rad = dep_array[:, 3]
    c_w_rad = dep_array[:, 4]
    c_W_rad = dep_array[:, 5]
    c_nu_rad = dep_array[:, 6]

    # Convert for basic plotting
    a_km = c_a_m / 1e3
    e = c_e
    i_deg = np.rad2deg(c_i_rad)
    w_deg = np.rad2deg(c_w_rad)
    W_deg = np.rad2deg(c_W_rad)
    nu_deg = np.rad2deg(c_nu_rad)

    # Physical/Propulsion (Cols 7-12)
    mass = dep_array[:, 7]
    accel_norm = dep_array[:, 8]
    thrust_force = accel_norm * mass
    alpha_rad_hist = dep_array[:, 9]
    beta_rad_hist = dep_array[:, 10]
    alpha_deg = np.rad2deg(alpha_rad_hist)
    beta_deg = np.rad2deg(beta_rad_hist)
    aspect_angle = np.rad2deg(dep_array[:, 11])

    # Q-Values (Cols 12-17)
    q_tot = dep_array[:, 12]
    q_a = dep_array[:, 13]
    q_e = dep_array[:, 14]
    q_i = dep_array[:, 15]
    q_w = dep_array[:, 16]
    q_W = dep_array[:, 17]

    # Target & Stage History (Cols 18-24)
    t_a_m = dep_array[:, 18]
    t_e = dep_array[:, 19]
    t_i_rad = dep_array[:, 20]
    t_w_rad = dep_array[:, 21]
    t_W_rad = dep_array[:, 22]
    t_nu_rad = dep_array[:, 23]
    target_kep_history = np.column_stack(
        (t_a_m, t_e, t_i_rad, t_w_rad, t_W_rad, t_nu_rad)
    )
    stage_history = dep_array[:, 24]

    # Eclipse
    shadow_factor = dep_array[:, 37]

    # Errors
    err_a = dep_array[:, 38]
    err_e = dep_array[:, 39]
    err_u = dep_array[:, 40]

    # Flags for plotting decisions
    is_rdv = sim_config.is_rendezvous

    # Stage Switch Extraction (Only relevant for RDV)
    transition_indices = []
    if is_rdv:
        # Finds EVERY index where the stage increments (e.g., 1->2, 2->3)
        transition_indices = np.where(np.diff(stage_history) > 0)[0]

    # Target Cartesian Trajectory
    tx_km = np.zeros(len(time_days))
    ty_km = np.zeros(len(time_days))
    tz_km = np.zeros(len(time_days))
    tvx_ms = np.zeros(len(time_days))
    tvy_ms = np.zeros(len(time_days))
    tvz_ms = np.zeros(len(time_days))

    for i in range(len(time_days)):
        cart = element_conversion.keplerian_to_cartesian(
            target_kep_history[i, :], DEFAULT_MU_EARTH
        )
        tx_km[i] = cart[0] / 1e3
        ty_km[i] = cart[1] / 1e3
        tz_km[i] = cart[2] / 1e3
        tvx_ms[i] = cart[3]
        tvy_ms[i] = cart[4]
        tvz_ms[i] = cart[5]

    # Fixed Target & Weights
    t = sim_config.target_state.values
    weights = sim_config.guidance.weights

    # ==========================================
    # 1. SLOW ORBITAL ELEMENTS (Osculating vs Mean)
    # ==========================================
    if log_cfg.plot_elements:

        # EXTRACT THE LIVE MEAN ELEMENTS DIRECTLY FROM THE INTEGRATOR PIPELINE
        c_mean_a = dep_array[:, 25]
        c_mean_e = dep_array[:, 26]
        c_mean_i = dep_array[:, 27]
        c_mean_w = dep_array[:, 28]
        c_mean_W = dep_array[:, 29]

        # Compute Target Means if dynamic
        t_mean_a = dep_array[:, 31]
        t_mean_e = dep_array[:, 32]
        t_mean_i = dep_array[:, 33]
        t_mean_w = dep_array[:, 34]
        t_mean_W = dep_array[:, 35]

        t_osc_vals = [
            target_kep_history[:, 0] / 1e3,
            target_kep_history[:, 1],
            np.rad2deg(target_kep_history[:, 2]),
            np.rad2deg(target_kep_history[:, 3]),
            np.rad2deg(target_kep_history[:, 4]),
        ]
        t_mean_vals = [
            t_mean_a / 1e3,
            t_mean_e,
            np.rad2deg(t_mean_i),
            np.rad2deg(t_mean_w),
            np.rad2deg(t_mean_W),
        ]

        fig_oe = make_subplots(
            rows=5,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(
                "SMA (km)",
                "Eccentricity",
                "Inclination (deg)",
                "Arg Peri (deg)",
                "RAAN (deg)",
            ),
        )

        elements = [
            (a_km, c_mean_a / 1e3, "SMA"),
            (e, c_mean_e, "Eccentricity"),
            (i_deg, np.rad2deg(c_mean_i), "Inclination"),
            (w_deg, np.rad2deg(c_mean_w), "Arg Peri"),
            (W_deg, np.rad2deg(c_mean_W), "RAAN"),
        ]

        colors = ["blue", "orange", "green", "purple", "brown"]

        for idx, (chaser_osc, chaser_mean, name) in enumerate(elements, start=1):
            color = colors[idx - 1]
            show_leg = idx == 1

            # 1. Chaser Osculating (Linked via legendgroup)
            fig_oe.add_trace(
                go.Scatter(
                    x=time_days,
                    y=chaser_osc,
                    mode="lines",
                    name="Chaser Osc",
                    line=dict(color=color, width=1),
                    showlegend=show_leg,
                    legendgroup="Chaser Osc",
                ),
                row=idx,
                col=1,
            )

            # 2. Chaser Mean (Linked via legendgroup)
            fig_oe.add_trace(
                go.Scatter(
                    x=time_days,
                    y=chaser_mean,
                    mode="lines",
                    name="Chaser Mean",
                    line=dict(color=color, dash="dash", width=2.5),
                    showlegend=show_leg,
                    legendgroup="Chaser Mean",
                ),
                row=idx,
                col=1,
            )

            # Target Lines
            # 3. Target Osculating
            fig_oe.add_trace(
                go.Scatter(
                    x=time_days,
                    y=t_osc_vals[idx - 1],
                    mode="lines",
                    name="Target Osc",
                    line=dict(color="red", width=1),
                    showlegend=show_leg,
                    legendgroup="Target Osc",
                ),
                row=idx,
                col=1,
            )

            # 4. Target Mean
            fig_oe.add_trace(
                go.Scatter(
                    x=time_days,
                    y=t_mean_vals[idx - 1],
                    mode="lines",
                    name="Target Mean",
                    line=dict(color="red", dash="dash", width=2.5),
                    showlegend=show_leg,
                    legendgroup="Target Mean",
                ),
                row=idx,
                col=1,
            )

            # Stage markers (RDV only)
            if is_rdv:
                for t_idx in transition_indices:
                    s_day = time_days[t_idx]
                    new_stage = int(stage_history[t_idx + 1])

                    fig_oe.add_vline(
                        x=s_day,
                        line_dash="dot",
                        line_color="gray",
                        opacity=0.8,
                        row=idx,
                        col=1,
                    )
                    fig_oe.add_trace(
                        go.Scatter(
                            x=[s_day],
                            y=[chaser_osc[t_idx]],
                            mode="markers",
                            marker=dict(color="red", size=8),
                            name=f"Stage {new_stage} Start",
                            showlegend=show_leg,
                            legendgroup="Stage Switch",
                            hovertemplate=f"Stage {new_stage}<extra></extra>",
                        ),
                        row=idx,
                        col=1,
                    )

        fig_oe.update_layout(
            title=f"Osculating vs. Mean Orbital Elements: {sim_config.name}",
            hovermode="x unified",
            height=1200,
            template="plotly_white",
        )
        fig_oe.update_xaxes(title_text="Time (Days)", row=5, col=1)
        fig_oe.update_yaxes(exponentformat="power", showexponent="all")
        fig_oe.show()

        # TRUE ANOMALY & PHASING (Strictly an RDV plot)
        if is_rdv:
            fig_nu = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=("Arg of Latitude (deg)", "Δ Latitude (deg)"),
            )

            chaser_u_wrap = (w_deg + nu_deg) % 360.0
            target_u_wrap = (
                np.rad2deg(target_kep_history[:, 3])
                + np.rad2deg(target_kep_history[:, 5])
            ) % 360.0
            delta_u = (chaser_u_wrap - target_u_wrap + 180.0) % 360.0 - 180.0

            fig_nu.add_trace(
                go.Scatter(
                    x=time_days,
                    y=chaser_u_wrap,
                    mode="lines",
                    name="Chaser",
                    line=dict(color="blue"),
                ),
                row=1,
                col=1,
            )
            fig_nu.add_trace(
                go.Scatter(
                    x=time_days,
                    y=target_u_wrap,
                    mode="lines",
                    name="Target",
                    line=dict(color="red"),
                ),
                row=1,
                col=1,
            )
            fig_nu.add_trace(
                go.Scatter(
                    x=time_days,
                    y=delta_u,
                    mode="lines",
                    name="Δ Arg of Latitude",
                    line=dict(color="black"),
                ),
                row=2,
                col=1,
            )
            fig_nu.add_hline(y=0, line_color="red", opacity=0.7, row=2, col=1)

            for t_idx in transition_indices:
                s_day = time_days[t_idx]
                for r in [1, 2]:
                    fig_nu.add_vline(
                        x=s_day,
                        line_dash="dot",
                        line_color="gray",
                        opacity=0.8,
                        row=r,
                        col=1,
                    )
                fig_nu.add_trace(
                    go.Scatter(
                        x=[s_day],
                        y=[chaser_u_wrap[t_idx]],
                        mode="markers",
                        marker=dict(color="red", size=8),
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )
                fig_nu.add_trace(
                    go.Scatter(
                        x=[s_day],
                        y=[delta_u[t_idx]],
                        mode="markers",
                        marker=dict(color="red", size=8),
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )

            fig_nu.update_layout(
                title="Phasing Evolution",
                hovermode="x unified",
                height=600,
                template="plotly_white",
            )
            fig_nu.update_xaxes(title_text="Time (Days)", row=2, col=1)
            fig_nu.update_yaxes(exponentformat="power", showexponent="all")
            fig_nu.show()

    # ==========================================
    # 2. Q HISTORY & 3. Q TERMS
    # ==========================================
    if log_cfg.plot_q_history:
        fig_q = make_subplots(specs=[[{"secondary_y": True}]])
        fig_q.add_trace(
            go.Scatter(
                x=time_days, y=q_tot, name="Q", line=dict(color="purple", width=2)
            ),
            secondary_y=False,
        )
        fig_q.add_trace(
            go.Scatter(
                x=time_days,
                y=np.sqrt(q_tot),
                name="sqrt(Q)",
                line=dict(color="teal", width=2),
            ),
            secondary_y=True,
        )
        fig_q.update_yaxes(
            title_text="Q", type="log", secondary_y=False, color="purple"
        )
        fig_q.update_yaxes(title_text="sqrt(Q)", secondary_y=True, color="teal")
        fig_q.update_layout(
            title="Lyapunov Function (Q) vs Time",
            hovermode="x unified",
            xaxis_title="Time (Days)",
            template="plotly_white",
        )
        fig_q.update_yaxes(exponentformat="power", showexponent="all")
        fig_q.show()

    if log_cfg.plot_q_terms:
        fig_q_terms = go.Figure()
        terms = [
            ("Q_a (SMA)", q_a),
            ("Q_e (Ecc)", q_e),
            ("Q_i (Inc)", q_i),
            ("Q_w (ArgP)", q_w),
            ("Q_W (RAAN)", q_W),
        ]
        for i, (label, data) in enumerate(terms):
            if weights[i] > 0:
                fig_q_terms.add_trace(
                    go.Scatter(x=time_days, y=data, mode="lines", name=label)
                )
        fig_q_terms.update_yaxes(
            type="log",
            title_text="Q Term Value",
            exponentformat="power",
            showexponent="all",
        )
        fig_q_terms.update_layout(
            title="Individual Q Terms vs Time",
            xaxis_title="Time (Days)",
            hovermode="x unified",
            template="plotly_white",
        )
        fig_q_terms.show()

    # ==========================================
    # 4. CONTROL, ASPECT, EFFICIENCY, ECLIPSE
    # ==========================================
    if log_cfg.plot_control_angles:
        fig_ctrl = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=("Alpha [deg]", "Beta [deg]"),
        )
        fig_ctrl.add_trace(
            go.Scatter(x=time_days, y=alpha_deg, line=dict(color="teal"), name="Alpha"),
            row=1,
            col=1,
        )
        fig_ctrl.add_trace(
            go.Scatter(x=time_days, y=beta_deg, line=dict(color="navy"), name="Beta"),
            row=2,
            col=1,
        )
        fig_ctrl.update_layout(
            title="Control Angles Evolution",
            hovermode="x unified",
            height=600,
            template="plotly_white",
        )
        fig_ctrl.update_xaxes(title_text="Time (Days)", row=2, col=1)
        fig_ctrl.update_yaxes(exponentformat="power", showexponent="all")
        fig_ctrl.show()

    if log_cfg.plot_aspect_angle:
        fig_aspect = go.Figure(
            go.Scatter(
                x=time_days, y=aspect_angle, line=dict(color="darkorange", width=2)
            )
        )
        fig_aspect.update_layout(
            title="Aspect Angle Evolution",
            xaxis_title="Time (Days)",
            yaxis_title="Aspect Angle [deg]",
            yaxis=dict(range=[0, 90]),
            template="plotly_white",
        )
        fig_aspect.update_yaxes(exponentformat="power", showexponent="all")
        fig_aspect.show()

    if log_cfg.plot_eclipse:
        fig_shadow = go.Figure(
            go.Scatter(
                x=time_days,
                y=shadow_factor,
                fill="tozeroy",
                mode="lines",
                line=dict(color="black", shape="hv", width=1.5),
                name="Sunlight Fraction",
            )
        )
        fig_shadow.update_layout(
            title="Eclipse History (Conical Shadow Factor)",
            xaxis_title="Time (Days)",
            yaxis_title="Shadow Factor (1 = Sun, 0 = Umbra)",
            yaxis=dict(
                range=[-0.1, 1.1], tickvals=[0, 1], ticktext=["Eclipse", "Sunlight"]
            ),
            template="plotly_white",
            height=300,
        )
        fig_shadow.show()

    # ==========================================
    # 5. MASS AND THRUST
    # ==========================================
    if log_cfg.plot_mass:
        fig_mass = go.Figure(
            go.Scatter(x=time_days, y=mass, line=dict(color="black", width=2))
        )
        fig_mass.update_layout(
            title="Spacecraft Mass vs Time",
            xaxis_title="Time (Days)",
            yaxis_title="Mass (kg)",
            template="plotly_white",
        )
        fig_mass.update_yaxes(exponentformat="power", showexponent="all")
        fig_mass.show()

    if log_cfg.plot_thrust:
        fig_thrust = go.Figure(
            go.Scatter(
                x=time_days,
                y=thrust_force,
                fill="tozeroy",
                line=dict(color="red", width=1.5),
            )
        )
        fig_thrust.update_layout(
            title="Thrust Magnitude vs Time",
            xaxis_title="Time (Days)",
            yaxis_title="Thrust Force (N)",
            template="plotly_white",
        )
        fig_thrust.update_yaxes(exponentformat="power", showexponent="all")
        fig_thrust.show()

    # ==========================================
    # 6. & 7. 3D TRAJECTORIES
    # ==========================================
    if log_cfg.plot_trajectory_3d:
        fig_3d = go.Figure()
        fig_3d.add_trace(
            go.Scatter3d(
                x=x_km,
                y=y_km,
                z=z_km,
                mode="lines",
                line=dict(color="blue", width=2),
                name="Chaser Trajectory",
            )
        )

        if is_rdv:
            fig_3d.add_trace(
                go.Scatter3d(
                    x=tx_km,
                    y=ty_km,
                    z=tz_km,
                    mode="lines",
                    line=dict(color="red", width=2),
                    name="Target Trajectory",
                )
            )

        if is_rdv:
            for t_idx in transition_indices:
                new_stage = int(stage_history[t_idx + 1])
                fig_3d.add_trace(
                    go.Scatter3d(
                        x=[x_km[t_idx]],
                        y=[y_km[t_idx]],
                        z=[z_km[t_idx]],
                        mode="markers",
                        marker=dict(color="magenta", size=6, symbol="diamond"),
                        name=f"Stage {new_stage} Start",
                    )
                )
            fig_3d.add_trace(
                go.Scatter3d(
                    x=[x_km[-1]],
                    y=[y_km[-1]],
                    z=[z_km[-1]],
                    mode="markers",
                    marker=dict(color="red", size=5, symbol="circle"),
                    name="Rendezvous Complete",
                )
            )

        r_earth = 6378.137
        u_surf, v_surf = np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 20j]
        fig_3d.add_trace(
            go.Surface(
                x=r_earth * np.cos(u_surf) * np.sin(v_surf),
                y=r_earth * np.sin(u_surf) * np.sin(v_surf),
                z=r_earth * np.cos(v_surf),
                opacity=0.3,
                showscale=False,
                colorscale="Greens",
                name="Earth",
            )
        )

        max_val = np.max(np.abs([x_km, y_km, z_km]))
        fig_3d.update_layout(
            title="3D Trajectory",
            scene=dict(
                xaxis=dict(range=[-max_val, max_val]),
                yaxis=dict(range=[-max_val, max_val]),
                zaxis=dict(range=[-max_val, max_val]),
                aspectmode="cube",
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            template="plotly_white",
        )
        fig_3d.show()

    if log_cfg.plot_rendezvous_zoom_3d and is_rdv:
        fig_zoom = go.Figure()
        idx_start = max(0, len(x_km) - 200)
        fig_zoom.add_trace(
            go.Scatter3d(
                x=x_km[idx_start:],
                y=y_km[idx_start:],
                z=z_km[idx_start:],
                mode="lines",
                line=dict(color="blue", width=3),
                name="Chaser Approach",
            )
        )
        fig_zoom.add_trace(
            go.Scatter3d(
                x=tx_km[idx_start:],
                y=ty_km[idx_start:],
                z=tz_km[idx_start:],
                mode="lines",
                line=dict(color="red", width=3),
                name="Target Path",
            )
        )
        fig_zoom.add_trace(
            go.Scatter3d(
                x=[x_km[-1]],
                y=[y_km[-1]],
                z=[z_km[-1]],
                mode="markers",
                marker=dict(color="blue", size=6, symbol="circle"),
                name="Chaser Final Position",
            )
        )
        fig_zoom.add_trace(
            go.Scatter3d(
                x=[tx_km[-1]],
                y=[ty_km[-1]],
                z=[tz_km[-1]],
                mode="markers",
                marker=dict(color="red", size=8, symbol="diamond"),
                name="Target Final Position",
            )
        )

        box_radius = 50.0
        tx, ty, tz = tx_km[-1], ty_km[-1], tz_km[-1]
        fig_zoom.update_layout(
            title="Rendezvous Final Approach (100km window)",
            scene=dict(
                xaxis=dict(range=[tx - box_radius, tx + box_radius]),
                yaxis=dict(range=[ty - box_radius, ty + box_radius]),
                zaxis=dict(range=[tz - box_radius, tz + box_radius]),
                aspectmode="cube",
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            template="plotly_white",
        )
        fig_zoom.show()

    # ==========================================
    # 8. RENDEZVOUS CONVERGENCE
    # ==========================================
    if log_cfg.plot_rendezvous_convergence and is_rdv:
        rel_pos = state_array[:, 1:4] - (np.column_stack((tx_km, ty_km, tz_km)) * 1e3)
        rel_vel = state_array[:, 4:7] - np.column_stack((tvx_ms, tvy_ms, tvz_ms))

        dist_km = np.linalg.norm(rel_pos, axis=1) / 1000.0
        vel_ms = np.linalg.norm(rel_vel, axis=1)

        fig_conv = make_subplots(specs=[[{"secondary_y": True}]])

        # Plot Distance
        fig_conv.add_trace(
            go.Scatter(
                x=time_days,
                y=dist_km,
                mode="lines",
                name="Distance",
                line=dict(color="blue", width=2),
            ),
            secondary_y=False,
        )

        # Plot Velocity
        fig_conv.add_trace(
            go.Scatter(
                x=time_days,
                y=vel_ms,
                mode="lines",
                name="Velocity",
                line=dict(color="red", width=2),
            ),
            secondary_y=True,
        )

        # --- DYNAMIC THRESHOLDS ---
        # Identify if the mission includes a final Cartesian proximity stage (Stage 3 or Baseline 0)
        final_stage = sim_config.stages[-1] if sim_config.stages else 0

        if final_stage == 3 or final_stage == 0:
            # Use the new configuration attributes
            dist_tol = getattr(sim_config, "stage_3_tol_distance", 10.0)
            vel_tol = getattr(sim_config, "stage_3_tol_velocity", 10.0)

            fig_conv.add_hline(
                y=np.log10(dist_tol),  # <--- THE FIX: np.log10()
                line_dash="dash",
                line_color="blue",
                opacity=0.5,
                secondary_y=False,
                annotation_text=f"Target Dist: {dist_tol} km",
                annotation_position="bottom right",
            )
            fig_conv.add_hline(
                y=np.log10(vel_tol),  # <--- THE FIX: np.log10()
                line_dash="dash",
                line_color="red",
                opacity=0.5,
                secondary_y=True,
                annotation_text=f"Target Vel: {vel_tol} m/s",
                annotation_position="top right",
            )

        # --- DYNAMIC STAGE TRANSITION MARKERS ---
        # Find every index where the stage number increases
        transition_indices = np.where(np.diff(stage_history) > 0)[0]

        for idx in transition_indices:
            switch_time = time_days[idx]
            new_stage = int(stage_history[idx + 1])

            # Vertical line for the transition
            fig_conv.add_vline(
                x=switch_time, line_dash="dot", line_color="gray", opacity=0.8
            )
            # Marker on the Distance line
            fig_conv.add_trace(
                go.Scatter(
                    x=[switch_time],
                    y=[dist_km[idx]],
                    mode="markers",
                    marker=dict(color="gray", size=8, symbol="diamond"),
                    showlegend=False,
                    name=f"Stage {new_stage} Start",
                    hovertemplate=f"Stage {new_stage} Start<br>Dist: %{{y:.2f}} km<extra></extra>",
                ),
                secondary_y=False,
            )
            # Marker on the Velocity line
            fig_conv.add_trace(
                go.Scatter(
                    x=[switch_time],
                    y=[vel_ms[idx]],
                    mode="markers",
                    marker=dict(color="gray", size=8, symbol="diamond"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                secondary_y=True,
            )

        fig_conv.update_yaxes(
            title_text="Relative Distance [km]",
            type="log",
            color="blue",
            secondary_y=False,
        )
        fig_conv.update_yaxes(
            title_text="Relative Velocity [m/s]",
            type="log",
            color="red",
            secondary_y=True,
        )
        fig_conv.update_layout(
            title=f"Rendezvous Capture Convergence ({sim_config.name})",
            hovermode="x unified",
            xaxis_title="Time [days]",
            template="plotly_white",
        )
        fig_conv.update_yaxes(exponentformat="power", showexponent="all")
        fig_conv.show()


    # ==========================================
    # 10. GVE & THRUST AUTHORITY
    # ==========================================
    if log_cfg.plot_thrust_components:
        from guidance.math_utils.shared_math import fast_gve_matrix_rtn

        dot_elements = np.zeros((len(time_days), 6))

        # Vector Unit Components (Direction Cosines)
        ur = np.cos(alpha_rad_hist) * np.cos(beta_rad_hist)
        ut = np.cos(alpha_rad_hist) * np.sin(beta_rad_hist)
        un = np.sin(alpha_rad_hist)

        # Squared Effectiveness (sums to exactly 100%)
        fr_pct = (ur**2) * 100
        ft_pct = (ut**2) * 100
        fn_pct = (un**2) * 100

        for k in range(len(time_days)):
            kep_k = np.array(
                [
                    a_km[k] * 1e3,
                    e[k],
                    np.deg2rad(i_deg[k]),
                    np.deg2rad(w_deg[k]),
                    np.deg2rad(W_deg[k]),
                    np.deg2rad(nu_deg[k]),
                ]
            )
            f_rtn = accel_norm[k] * np.array([ur[k], ut[k], un[k]])
            B = fast_gve_matrix_rtn(kep_k, DEFAULT_MU_EARTH)
            dot_elements[k, :] = B @ f_rtn

        # Convert Rates to Mission-Relevant Units
        da_km_day = dot_elements[:, 0] * 86400 / 1000.0  # m/s -> km/day
        de_per_day = dot_elements[:, 1] * 86400  # 1/s -> 1/day
        di_deg_day = np.rad2deg(dot_elements[:, 2]) * 86400  # rad/s -> deg/day

        fig_gve = make_subplots(
            rows=3,
            cols=2,
            column_titles=["Thrust Effort Allocation (%)", "Physical Mission Rates"],
            vertical_spacing=0.07,
            shared_xaxes=True,
        )

        fig_gve.add_trace(
            go.Scatter(
                x=time_days, y=ft_pct, name="Tangential %", line=dict(color="blue")
            ),
            row=1,
            col=1,
        )
        fig_gve.add_trace(
            go.Scatter(
                x=time_days, y=da_km_day, name="da/dt [km/day]", line=dict(color="blue")
            ),
            row=1,
            col=2,
        )

        fig_gve.add_trace(
            go.Scatter(
                x=time_days, y=fr_pct, name="Radial %", line=dict(color="orange")
            ),
            row=2,
            col=1,
        )
        fig_gve.add_trace(
            go.Scatter(
                x=time_days,
                y=de_per_day,
                name="de/dt [1/day]",
                line=dict(color="orange"),
            ),
            row=2,
            col=2,
        )

        fig_gve.add_trace(
            go.Scatter(
                x=time_days, y=fn_pct, name="Normal %", line=dict(color="green")
            ),
            row=3,
            col=1,
        )
        fig_gve.add_trace(
            go.Scatter(
                x=time_days,
                y=di_deg_day,
                name="di/dt [deg/day]",
                line=dict(color="green"),
            ),
            row=3,
            col=2,
        )

        fig_gve.update_layout(
            title="Control Effort Distribution & Mission Progression",
            height=900,
            template="plotly_white",
            hovermode="x unified",
        )
        fig_gve.update_yaxes(title_text="Allocated %", row=2, col=1)
        fig_gve.update_yaxes(title_text="km / day", row=1, col=2)
        fig_gve.update_yaxes(title_text="Δe / day", row=2, col=2)
        fig_gve.update_yaxes(title_text="deg / day", row=3, col=2)
        fig_gve.update_yaxes(exponentformat="power", showexponent="all")
        fig_gve.show()

    if log_cfg.plot_errors:
        fig_err = go.Figure()

        # Plot all three independent normalized errors
        fig_err.add_trace(
            go.Scatter(
                x=time_days,
                y=err_a,
                mode="lines",
                name="Norm Error: SMA (a)",
                line=dict(color="blue", width=2),
            )
        )
        fig_err.add_trace(
            go.Scatter(
                x=time_days,
                y=err_e,
                mode="lines",
                name="Norm Error: Ecc (e)",
                line=dict(color="orange", width=2),
            )
        )

        # Use a dashed line for Phase error so it's easy to distinguish
        fig_err.add_trace(
            go.Scatter(
                x=time_days,
                y=err_u,
                mode="lines",
                name="Norm Error: Phase (u)",
                line=dict(color="purple", width=2, dash="dash"),
            )
        )

        # The Golden threshold line that triggers the next stage
        fig_err.add_hline(
            y=1.0,
            line_dash="solid",
            line_color="green",
            opacity=0.6,
            annotation_text="Transition Threshold (1.0)",
            annotation_position="top right",
        )

        # Plot vertical markers when stages switch
        for idx in transition_indices:
            switch_time = time_days[idx]
            new_stage = int(stage_history[idx + 1])

            fig_err.add_vline(
                x=switch_time, line_dash="dot", line_color="gray", opacity=0.8
            )

            # Find the maximum error at the moment of transition to place the diamond
            max_err_at_switch = max(err_a[idx], err_e[idx], err_u[idx])
            fig_err.add_trace(
                go.Scatter(
                    x=[switch_time],
                    y=[max_err_at_switch],
                    mode="markers",
                    marker=dict(color="red", size=10, symbol="diamond"),
                    name=f"Stage {new_stage} Triggered",
                    hovertemplate=f"Stage {new_stage} Triggered<extra></extra>",
                )
            )

        fig_err.update_yaxes(
            title_text="Normalized Error [-]",
            type="log",
            exponentformat="power",
            showexponent="all",
        )
        fig_err.update_layout(
            title=f"Detailed Stage Transition Errors ({sim_config.name})",
            hovermode="x unified",
            xaxis_title="Time [days]",
            template="plotly_white",
            height=500,
        )
        fig_err.show()
