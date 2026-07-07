import os
import json
import dataclasses
from enum import Enum
from datetime import datetime

import numpy as np
import pandas as pd
from numba import njit

from tudatpy.astro import element_conversion

from config import DEFAULT_MU_EARTH
from guidance.math_utils.shared_math import fast_gve_matrix_rtn


@njit(cache=True, fastmath=True)
def _fast_batch_gve_rates(
    time_sec, c_a, c_e, c_i, c_w, c_W, c_nu, accel_norm, ur, ut, un, mu
):
    n = len(time_sec)
    da, de, di = np.zeros(n), np.zeros(n), np.zeros(n)

    for k in range(n):
        kep = np.array([c_a[k], c_e[k], c_i[k], c_w[k], c_W[k], c_nu[k]])
        f_rtn = accel_norm[k] * np.array([ur[k], ut[k], un[k]])
        B = fast_gve_matrix_rtn(kep, mu)
        dot_elements = B @ f_rtn

        da[k] = dot_elements[0] * 86400 / 1000.0
        de[k] = dot_elements[1] * 86400
        di[k] = dot_elements[2] * 86400 * (180.0 / np.pi)

    return da, de, di


class SimulationEncoder(json.JSONEncoder):
    """Custom JSON Encoder to handle Python Dataclasses, Enums, and Numpy Arrays."""

    def default(self, obj):
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        if isinstance(obj, Enum):
            return obj.name
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def save_simulation_data(sim_config, state_array, dep_array, res_summary):
    """
    Saves the simulation config as JSON and the full telemetry history as a Parquet file.
    Includes automatic calculation of mean elements and relative R&PO states.
    """
    # 1. Create the dedicated results folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    folder_name = f"{timestamp}_{sim_config.name}"
    folder_path = os.path.join("results", folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # 2. Dump the Configuration & Results Summary to JSON
    config_file = os.path.join(folder_path, "config.json")

    # Merge the config and the summary for easy reference
    save_dict = {"summary": res_summary, "configuration": sim_config}
    with open(config_file, "w") as f:
        json.dump(save_dict, f, cls=SimulationEncoder, indent=4)

    # 3. Extract Raw Arrays
    time_sec = state_array[:, 0]
    time_days = (time_sec - time_sec[0]) / 86400.0

    c_a = dep_array[:, 1]
    c_e = dep_array[:, 2]
    c_i = dep_array[:, 3]
    c_w = dep_array[:, 4]
    c_W = dep_array[:, 5]
    c_nu = dep_array[:, 6]

    t_a = dep_array[:, 18]
    t_e = dep_array[:, 19]
    t_i = dep_array[:, 20]
    t_w = dep_array[:, 21]
    t_W = dep_array[:, 22]
    t_nu = dep_array[:, 23]
    c_mean_a = dep_array[:, 25]
    c_mean_e = dep_array[:, 26]
    c_mean_i = dep_array[:, 27]
    c_mean_w = dep_array[:, 28]
    c_mean_W = dep_array[:, 29]
    c_mean_nu_eff = dep_array[:, 30]
    t_mean_a = dep_array[:, 31]
    t_mean_e = dep_array[:, 32]
    t_mean_i = dep_array[:, 33]
    t_mean_w = dep_array[:, 34]
    t_mean_W = dep_array[:, 35]
    t_mean_nu_eff = dep_array[:, 36]

    # NEW SHADOW FACTOR
    shadow_factor = dep_array[:, 37]

    err_a = dep_array[:, 38]
    err_e = dep_array[:, 39]
    err_u = dep_array[:, 40]

    # 5. Compute Relative States (Convert Target Kep to Cart)
    c_cart = state_array[:, 1:7]
    t_cart = np.zeros_like(c_cart)

    for i in range(len(time_sec)):
        t_cart[i, :] = element_conversion.keplerian_to_cartesian(
            dep_array[i, 18:24], DEFAULT_MU_EARTH
        )

    rel_pos = c_cart[:, 0:3] - t_cart[:, 0:3]
    rel_vel = c_cart[:, 3:6] - t_cart[:, 3:6]
    rel_dist_km = np.linalg.norm(rel_pos, axis=1) / 1000.0
    rel_vel_ms = np.linalg.norm(rel_vel, axis=1)

    # 6. Argument of Latitude computation
    c_u = np.mod(c_w + c_nu, 2 * np.pi)
    t_u = np.mod(t_w + t_nu, 2 * np.pi)
    delta_u = np.mod(c_u - t_u + np.pi, 2 * np.pi) - np.pi

    # 6.5 Compute GVE & Thrust Authority Allocations
    alpha_rad = dep_array[:, 9]
    beta_rad = dep_array[:, 10]
    accel_norm = dep_array[:, 8]

    # Direction Cosines
    ur = np.cos(alpha_rad) * np.cos(beta_rad)
    ut = np.cos(alpha_rad) * np.sin(beta_rad)
    un = np.sin(alpha_rad)

    # Squared Effectiveness (%)
    fr_pct = (ur**2) * 100
    ft_pct = (ut**2) * 100
    fn_pct = (un**2) * 100

    # Physical Mission Rates via GVE
    da_km_day = np.zeros_like(time_sec)
    de_per_day = np.zeros_like(time_sec)
    di_deg_day = np.zeros_like(time_sec)

    da_km_day, de_per_day, di_deg_day = _fast_batch_gve_rates(
        time_sec,
        c_a,
        c_e,
        c_i,
        c_w,
        c_W,
        c_nu,
        accel_norm,
        ur,
        ut,
        un,
        DEFAULT_MU_EARTH,
    )

    # 7. Construct the Master DataFrame
    df = pd.DataFrame(
        {
            "time_days": time_days,
            "time_epoch_sec": time_sec,
            # Chaser Osculating
            "chaser_osc_a_m": c_a,
            "chaser_osc_e": c_e,
            "chaser_osc_i_rad": c_i,
            "chaser_osc_w_rad": c_w,
            "chaser_osc_W_rad": c_W,
            "chaser_osc_nu_rad": c_nu,
            "chaser_osc_u_rad": c_u,
            # Chaser Mean
            "chaser_mean_a_km": c_mean_a / 1000.0,
            "chaser_mean_e": c_mean_e,
            "chaser_mean_i_rad": c_mean_i,
            "chaser_mean_w_rad": c_mean_w,
            "chaser_mean_W_rad": c_mean_W,
            "chaser_mean_nu_eff_rad": c_mean_nu_eff,
            # Target Osculating
            "target_osc_a_m": t_a,
            "target_osc_e": t_e,
            "target_osc_i_rad": t_i,
            "target_osc_w_rad": t_w,
            "target_osc_W_rad": t_W,
            "target_osc_nu_rad": t_nu,
            "target_osc_u_rad": t_u,
            # Target Mean
            "target_mean_a_km": t_mean_a / 1000.0,
            "target_mean_e": t_mean_e,
            "target_mean_i_rad": t_mean_i,
            "target_mean_w_rad": t_mean_w,
            "target_mean_W_rad": t_mean_W,
            "target_mean_nu_eff_rad": t_mean_nu_eff,
            # Relative States
            "delta_u_rad": delta_u,
            "relative_distance_km": rel_dist_km,
            "relative_velocity_ms": rel_vel_ms,
            # Physical & Environmental
            "chaser_mass_kg": dep_array[:, 7],
            "thrust_accel_norm": dep_array[:, 8],
            "alpha_rad": dep_array[:, 9],
            "beta_rad": dep_array[:, 10],
            "aspect_angle_rad": dep_array[:, 11],
            "shadow_factor": shadow_factor,
            # Q-Law Logic & Terms
            "error_a_norm": err_a,
            "error_e_norm": err_e,
            "error_u_norm": err_u,
            "q_tot": dep_array[:, 12],
            "q_a": dep_array[:, 13],
            "q_e": dep_array[:, 14],
            "q_i": dep_array[:, 15],
            "q_w": dep_array[:, 16],
            "q_W": dep_array[:, 17],
            "stage": dep_array[:, 24],
            "thrust_pct_radial": fr_pct,
            "thrust_pct_tangential": ft_pct,
            "thrust_pct_normal": fn_pct,
            "rate_a_km_day": da_km_day,
            "rate_e_per_day": de_per_day,
            "rate_i_deg_day": di_deg_day,
        }
    )

    # 8. Save the Parquet file
    telemetry_file = os.path.join(folder_path, "telemetry.parquet")
    df.to_parquet(telemetry_file, index=False)

    print(
        f"\n[DATA LOGGER] Simulation data and telemetry successfully saved to: {folder_path}/"
    )
