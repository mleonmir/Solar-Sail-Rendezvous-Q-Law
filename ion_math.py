import numpy as np
from numba import njit

from guidance.math_utils.shared_math import fast_gve_matrix_rtn, wrap_to_pi


@njit(cache=True, fastmath=True)
def fast_q_gradient_ion(
    kep: np.ndarray,
    dynamic_target: np.ndarray,
    weights: np.ndarray,
    q_law_params: np.ndarray,
    max_oe_rates: np.ndarray,
    is_stage_2: bool,
) -> np.ndarray:
    """
    Computes the gradient of the Lyapunov function for Q-Law guidance in the context of ion propulsion.
    """

    # Unpack variables (Using dynamic_target to respect J2 drift!)
    a_sc, e_sc, i_sc, w_sc, W_sc, nu_sc = kep
    a_t, e_t, i_t, w_t, W_t, nu_t = dynamic_target

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

    d_e = e_sc - e_t
    d_i = i_sc - i_t
    d_w = wrap_to_pi(w_sc - w_t)
    d_W = wrap_to_pi(W_sc - W_t)

    # --- RDV logic based on argument of latitude ---
    if is_stage_2:
        u_chaser = w_sc + nu_sc
        u_target = w_t + nu_t
        delta_u = wrap_to_pi(u_chaser - u_target)

        # Call the helper function!
        a_T, da_aug_de, da_aug_du = compute_augmented_target_and_derivative(
            a_t, e_sc, delta_u, W_L, W_scl, rp_min
        )

    d_a = a_sc - a_T

    # Penalty Term
    rp = a_sc * (1.0 - e_sc)
    P_imp = w_imp * np.exp(k_imp * (1.0 - rp / rp_min))
    dP_imp_drp = -(k_imp / rp_min) * P_imp
    dP_imp_da = dP_imp_drp * (1.0 - e_sc)
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
        weights[0] * S_a * (abs(d_a) / max_a) ** 2
        + weights[1] * 1 * (abs(d_e) / max_e) ** 2
        + weights[2] * 1 * (abs(d_i) / max_i) ** 2
        + weights[3] * 1 * (abs(d_w) / max_w) ** 2
        + weights[4] * 1 * (abs(d_W) / max_W) ** 2
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

    if is_stage_2:
        grad_e -= (Penalty_Mult * dQ_term_a) * da_aug_de
        grad_w -= (Penalty_Mult * dQ_term_a) * da_aug_du

    return np.array([grad_a, grad_e, grad_i, grad_w, grad_W, grad_nu])


@njit(cache=True, fastmath=True)
def compute_max_rates_ion(
    kep: np.ndarray, thrust: float, mass: float, mu: float
) -> list:
    """
    Calculates the maximum rates of change for orbital elements given current thrust and mass.
    Args:
        - kep (np.ndarray): Current Keplerian elements.
        - thrust (float): Spacecraft thrust magnitude in Newtons.
        - mass (float): Current spacecraft mass in kg.
        - mu (float): Gravitational parameter of the central body.
    Returns:
        - list: Maximum rates [max_a, max_e, max_i, max_w, max_W].
    """
    # --- General Initialization ---
    a, e, i, w, _, _ = kep
    f_accel = thrust / mass
    p = a * (1 - e**2)
    h = np.sqrt(mu * p)

    # --- Safety clips to avoid division by zero ---
    safe_e = min(e, 0.999)
    safe_sin_i = max(1e-6, abs(np.sin(i)))

    # --- a ---
    term_a_num = a**3 * (1 + safe_e)
    term_a_den = mu * (1 - safe_e)
    max_a = 2 * f_accel * np.sqrt(term_a_num / term_a_den)

    # --- e ---
    max_e = 2 * p * f_accel / h

    # --- i ---
    denom_i = np.sqrt(max(0.0, 1.0 - safe_e**2 * np.sin(w) ** 2)) - safe_e * abs(
        np.cos(w)
    )
    max_i = (p * f_accel) / (h * max(1e-12, denom_i))

    # --- W ---
    denom_W = np.sqrt(max(0.0, 1.0 - safe_e**2 * np.cos(w) ** 2)) - safe_e * abs(
        np.sin(w)
    )
    max_W = (p * f_accel) / (h * safe_sin_i * max(1e-12, denom_W))

    # --- w ---
    safe_e_div = max(e, 1e-4)
    term1 = (1.0 - safe_e_div**2) / (2.0 * safe_e_div**3)
    term2 = np.sqrt(0.25 * ((1.0 - safe_e_div**2) / (safe_e_div**3)) ** 2 + 1.0 / 27.0)
    u = (term1 + term2) ** (1.0 / 3.0)
    v = (-term1 + term2) ** (1.0 / 3.0)
    cos_nu_xx = u - v - (1.0 / safe_e_div)
    cos_nu_xx = max(-1.0, min(1.0, cos_nu_xx))
    sin_nu_xx_sq = 1.0 - cos_nu_xx**2
    r_xx = p / (1.0 + safe_e_div * cos_nu_xx)
    max_w = (f_accel / (safe_e_div * h)) * np.sqrt(
        p**2 * cos_nu_xx**2 + (p + r_xx) ** 2 * sin_nu_xx_sq
    )

    return [max_a, max_e, max_i, max_w, max_W]


# Helper functions for gradient, max rates and efficiency computations
@njit(cache=True, fastmath=True)
def compute_augmented_target_and_derivative(
    a_target: float,
    e_chaser: float,
    delta_u: float,
    W_L: float,
    W_scl: float,
    rp_min: float,
) -> tuple:
    """
    Computes the augmented semi-major axis target for phase shifting during rendezvous.
    Args:
        - a_target (float): Original target semi-major axis.
        - e_chaser (float): Chaser's current eccentricity.
        - delta_u (float): Phase difference between chaser and target.
        - W_L (float): Phasing penalty weight.
        - W_scl (float): Phasing scaling factor.
        - rp_min (float): Minimum allowable periapsis radius.
    Returns:
        - tuple: Augmented SMA, and its derivatives w.r.t true anomaly and eccentricity.
    """
    delta_u = wrap_to_pi(delta_u)

    term_e = rp_min / (1.0 - e_chaser)
    K = a_target - term_e

    scaled_error = W_scl * delta_u
    atan_val = np.arctan(scaled_error)
    common_factor = 2.0 * W_L / np.pi

    a_target_aug = common_factor * K * atan_val

    # Calculate the derivatives
    da_aug_de = -common_factor * atan_val * (rp_min / (1.0 - e_chaser) ** 2)
    da_aug_du = common_factor * K * (W_scl / (1.0 + scaled_error**2))

    return a_target + a_target_aug, da_aug_de, da_aug_du
