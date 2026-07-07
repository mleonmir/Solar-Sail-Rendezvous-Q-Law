import numpy as np
from numba import njit

from guidance.math_utils.shared_math import fast_gve_matrix_rtn, wrap_to_pi

IDEAL = 0


@njit(cache=True, fastmath=True)
def fast_q_gradient_sail(
    kep: np.ndarray,
    dynamic_target: np.ndarray,
    weights: np.ndarray,
    q_law_params: np.ndarray,
    max_oe_rates: np.ndarray,
    is_phasing_active: bool,
) -> np.ndarray:
    """
    Computes the gradient of the Q-Law Lyapunov function for a solar sail.
    Args:
        kep (np.ndarray): Current Keplerian elements (6,).
        dynamic_target (np.ndarray): Dynamic target Keplerian elements (6,).
        weights (np.ndarray): Penalty weights for each element (6,).
        q_law_params (np.ndarray): Q-law scaling and penalty parameters.
        max_oe_rates (np.ndarray): Maximum rates of change for orbital elements (5,).
        is_phasing_active (bool): Flag indicating whether phasing is active.
    Returns:
        np.ndarray: Gradient of the Lyapunov function w.r.t [a, e, i, w, W] (5,).
    """
    # Unpack variables
    a_sc, e_sc, i_sc, w_sc, W_sc, nu_sc = kep
    a_t, e_t, i_t, w_t, W_t, nu_t = dynamic_target
    u_chaser = w_sc + nu_sc
    u_target = w_t + nu_t

    max_a = max_oe_rates[0] if abs(max_oe_rates[0]) > 1e-12 else 1e-12
    max_e = max_oe_rates[1] if abs(max_oe_rates[1]) > 1e-12 else 1e-12
    max_i = max_oe_rates[2] if abs(max_oe_rates[2]) > 1e-12 else 1e-12
    max_w = max_oe_rates[3] if abs(max_oe_rates[3]) > 1e-12 else 1e-12
    max_W = max_oe_rates[4] if abs(max_oe_rates[4]) > 1e-12 else 1e-12
    m, n, r = q_law_params[0], q_law_params[1], q_law_params[2]
    k_imp, rp_min, w_imp = q_law_params[3], q_law_params[4], q_law_params[5]
    k_esc, a_max, w_esc = q_law_params[6], q_law_params[7], q_law_params[8]
    W_L, W_scl = q_law_params[9], q_law_params[10]
    da_aug_du = 0.0
    da_aug_de = 0.0
    a_T = a_t

    # Distance calculation
    d_e = e_sc - e_t
    d_i = i_sc - i_t
    d_w = wrap_to_pi(w_sc - w_t)
    d_W = wrap_to_pi(W_sc - W_t)
    delta_u = wrap_to_pi(u_chaser - u_target)

    # --- RDV logic based on argument of latitude instead of true anomaly ---
    if is_phasing_active:
        a_T, da_aug_de, da_aug_du = compute_augmented_sma(
            a_t, e_sc, delta_u, W_L, W_scl, rp_min
        )

    d_a = a_sc - a_T

    # Penalty Term
    rp = a_sc * (1.0 - e_sc)
    P_imp = w_imp * np.exp(k_imp * (1.0 - rp / rp_min))
    dP_imp_drp = -(k_imp / rp_min) * P_imp
    dP_imp_da = dP_imp_drp * (1.0 - e_sc)  # dP_da = dP_drp * drp_da
    dP_imp_de = dP_imp_drp * (-a_sc)

    P_esc = w_esc * np.exp(k_esc * (a_sc - a_max) / a_max)
    dP_esc_da = (k_esc / a_max) * P_esc

    Penalty_Mult = 1 + P_imp + P_esc

    # Scaling Term
    term_a = (d_a / (m * a_T)) ** n
    S_a = (1.0 + term_a) ** (1.0 / r)
    if abs(d_a) > 1e-12:
        dS_da = (S_a / (r * (1.0 + term_a))) * n * (term_a / d_a)
    else:
        dS_da = 0.0

    # Sum of Q Terms
    Sum_Q = (
        weights[0] * S_a * (abs(d_a) / -max_a) ** 2
        + weights[1] * 1 * (abs(d_e) / -max_e) ** 2
        + weights[2] * 1 * (abs(d_i) / -max_i) ** 2
        + weights[3] * 1 * (abs(d_w) / -max_w) ** 2
        + weights[4] * 1 * (abs(d_W) / -max_W) ** 2
    )

    # Gradients
    dQ_term_a = (weights[0] / max_a**2) * (dS_da * d_a**2 + S_a * 2.0 * d_a)
    grad_a = ((dP_imp_da + dP_esc_da) * Sum_Q) + Penalty_Mult * dQ_term_a

    dQ_term_e = 2.0 * weights[1] * d_e / max_e**2
    grad_e = (dP_imp_de * Sum_Q) + Penalty_Mult * dQ_term_e

    grad_i = Penalty_Mult * (2.0 * weights[2] * d_i / max_i**2)

    grad_w = Penalty_Mult * (2.0 * weights[3] * d_w / max_w**2)

    grad_W = Penalty_Mult * (2.0 * weights[4] * d_W / max_W**2)

    grad_nu = 0.0

    # Grad update due to augmented SMA
    grad_e -= (Penalty_Mult * dQ_term_a) * da_aug_de
    grad_w -= (Penalty_Mult * dQ_term_a) * da_aug_du

    return np.array([grad_a, grad_e, grad_i, grad_w, grad_W, grad_nu])


@njit(cache=True, fastmath=True)
def compute_augmented_sma(
    a_target: float,
    e_chaser: float,
    u_error: float,
    W_L: float,
    W_scl: float,
    rp_min: float,
) -> tuple:
    """Generic augmented SMA calculator for in-plane phasing."""

    term_e = rp_min / (1.0 - e_chaser)
    K = a_target - term_e

    scaled_error = W_scl * u_error
    atan_val = np.arctan(scaled_error)
    common_factor = 2.0 * W_L / np.pi

    a_target_aug = common_factor * K * atan_val
    da_aug_de = -common_factor * atan_val * (rp_min / (1.0 - e_chaser) ** 2)
    da_aug_du = common_factor * K * (W_scl / (1.0 + scaled_error**2))

    return a_target + a_target_aug, da_aug_de, da_aug_du


@njit(cache=True, fastmath=True)
def compute_optimal_angles_sail(
    primer_s: np.ndarray, model_id: int, model_params: np.ndarray
) -> tuple:
    """
    Computes optimal pitch (alpha) and clock (beta) angles for the solar sail.
    Args:
        - primer_s (np.ndarray): Primer vector in the Sun-pointing frame (3,).
        - model_id (int): Identifier for the sail force model (IDEAL or others).
        - model_params (np.ndarray): Parameters for the optical model (C1, C2, C3).
    Returns:
        - tuple: (alpha_star, beta_star) optimal angles in radians.
    """
    px, py, pz = primer_s[0], primer_s[1], primer_s[2]
    
    beta_star = np.arctan2(py, pz)
    
    if model_id == IDEAL:
        term1 = 2.0 * np.sqrt(py**2+pz**2)
        term2 = 3.0 * px + np.sqrt(9.0 * px**2 + 8.0 * (py**2+pz**2))
        alpha_star = np.arctan2(term1, term2)
        
    else:
        raise ValueError("Invalid model_id provided")

    return alpha_star, beta_star


@njit(cache=True, fastmath=True)
def get_acceleration_vector_s(
    alpha_star: float,
    beta_star: float,
    dist_sun: float,
    area_to_mass: float,
    model_id: int,
    P_1AU: float,
    AU: float,
) -> np.ndarray:
    """
    Computes the resulting acceleration vector in the Sun-pointing frame.
    Args:
        - alpha_star (float): Pitch angle in radians.
        - beta_star (float): Clock angle in radians.
        - dist_sun (float): Distance from the Sun in meters.
        - area_to_mass (float): Area-to-mass ratio of the sail.
        - model_id (int): Sail force model identifier.
        - P_1AU (float): Solar radiation pressure at 1 AU.
        - AU (float): Astronomical Unit in meters.
    Returns:
        - np.ndarray: Acceleration vector in the Sun-pointing frame (3,).
    """
    c_alpha = np.cos(alpha_star)
    s_alpha = np.sin(alpha_star)
    c_beta = np.cos(beta_star)
    s_beta = np.sin(beta_star)

    pressure = P_1AU * (AU**2 / dist_sun**2)
    normal_s = np.array([c_alpha, s_alpha * s_beta, s_alpha * c_beta])

    if model_id == IDEAL:
        a_s = 2.0 * area_to_mass * pressure * c_alpha**2 * normal_s

    else:
        raise ValueError("Invalid model_id provided to get_acceleration_vector_s")

    return a_s


@njit(cache=True, fastmath=True)
def compute_max_rates_sail_triple_min(
    P_1AU: float, area_to_mass: float, initial_state: np.ndarray, mu_earth: float
) -> tuple:
    """
    Computes the constant conservative maximum rates of change for a sail (Triple Min approx).
    Args:
        - P_1AU (float): Solar radiation pressure at 1 AU.
        - area_to_mass (float): Area-to-mass ratio of the sail.
        - initial_state (np.ndarray): Initial Keplerian elements (6,).
        - mu_earth (float): Earth's gravitational parameter.
    Returns:
        -tuple: Max rates (max_a, max_e, max_i, max_w, max_W).
    """
    sma0, e0, i0, w0, _, _ = initial_state

    a0 = 2.0 * P_1AU * area_to_mass
    p0 = sma0 * (1.0 - e0**2)
    h0 = np.sqrt(mu_earth * p0)
    alpha_opt = np.arcsin(1.0 / np.sqrt(3.0))  # ~35.26 deg
    f_radial_max = a0
    f_transverse_max = a0 * (np.cos(alpha_opt) ** 2) * np.sin(alpha_opt)

    max_a = (2.0 * sma0**2 / h0) * f_radial_max 

    max_e = (2.0 * p0 / h0) * f_transverse_max

    denom_i = np.sqrt(max(0.0, 1.0 - e0**2 * np.sin(w0) ** 2)) - e0 * abs(np.cos(w0))
    max_i = (p0 / h0) * f_transverse_max / max(1e-12, denom_i)

    safe_sin_i = max(1e-6, abs(np.sin(i0)))
    denom_W = np.sqrt(max(0.0, 1.0 - e0**2 * np.cos(w0) ** 2)) - e0 * abs(np.sin(w0))
    max_W = (p0 / h0) * f_transverse_max / (safe_sin_i * max(1e-12, denom_W))

    safe_e = max(e0, 1e-4)
    max_w = 2 * p0 * (f_transverse_max / (safe_e * h0))

    return max_a, max_e, max_i, max_w, max_W
