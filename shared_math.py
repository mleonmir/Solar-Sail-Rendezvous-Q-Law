import numpy as np
from numba import njit

from tudatpy.astro import element_conversion
from tudatpy.interface import spice

from config import CoordinateType, ReferenceFrame, StateDefinition

# =============================================================================
# Fast math calculation using Numba to speed simulation
# =============================================================================


# Frame transformation matrices
@njit(cache=True, fastmath=True)
def fast_rot_mat_eci2rtn(r_eci: np.ndarray, v_eci: np.ndarray) -> np.ndarray:
    """
    Computes rotation matrix from ECI to Radial-Transverse-Normal (RTN) frame.
    Args:
        - r_eci (np.ndarray): Position vector in ECI frame.
        - v_eci (np.ndarray): Velocity vector in ECI frame.
    Returns:
        - np.ndarray: 3x3 rotation matrix.
    """
    u_r = r_eci / np.linalg.norm(r_eci)
    h_vec = np.cross(r_eci, v_eci)
    u_n = h_vec / np.linalg.norm(h_vec)
    u_t = np.cross(u_n, u_r)

    return np.vstack((u_r, u_t, u_n))


@njit(cache=True, fastmath=True)
def fast_rot_mat_p2rtn(nu: float) -> np.ndarray:
    """
    Computes rotation matrix from Perifocal (Orbit) to Radial-Transverse-Normal (RTN) frame.
    Args:
        - nu (float): True anomaly in radians.
    Returns:
        - np.ndarray: 3x3 rotation matrix.
    """
    c_v, s_v = np.cos(nu), np.sin(nu)
    return np.array([[c_v, s_v, 0.0], [-s_v, c_v, 0.0], [0.0, 0.0, 1.0]])


@njit(cache=True, fastmath=True)
def fast_rot_mat_eci2s(s_vec_eci: np.ndarray, R_eci2ecl: np.ndarray) -> np.ndarray:
    """
    Computes rotation matrix from ECI to Sun-Fixed frame.
    Args:
        - s_vec_eci (np.ndarray): Unit vector pointing from Spacecraft to Sun in ECI frame.
        - R_eci2ecl (np.ndarray): Rotation matrix from ECI to ECLIPJ2000 frame.
    Returns:
        - np.ndarray: 3x3 rotation matrix.
    """
    # 1. Transform sun vector to ecliptic
    x_s_ecl = R_eci2ecl @ s_vec_eci

    # 2. Build S-frame in ecliptic coordinates
    z_ref = np.array([0.0, 0.0, 1.0])
    y_s_ecl = np.cross(z_ref, x_s_ecl)
    y_s_norm = np.linalg.norm(y_s_ecl)

    # Handle singularity if sun is at ecliptic pole
    if y_s_norm < 1e-12:
        y_s_ecl = np.array([0.0, 1.0, 0.0])
    else:
        y_s_ecl /= y_s_norm

    z_s_ecl = np.cross(x_s_ecl, y_s_ecl)

    # 3. Assemble R_ecl2s and combine with ECI2ECL
    R_ecl2s = np.vstack((x_s_ecl, y_s_ecl, z_s_ecl))
    return R_ecl2s @ R_eci2ecl


@njit(cache=True, fastmath=True)
def fast_rot_mat_eci2p(r_eci: np.ndarray, v_eci: np.ndarray, nu: float) -> np.ndarray:
    """
    Computes rotation matrix from ECI to Perifocal (orbit) frame.
    Args:
        - r_eci (np.ndarray): Position vector in ECI frame.
        - v_eci (np.ndarray): Velocity vector in ECI frame.
        - nu (float): True anomaly in radians.
    Returns:
        - np.ndarray: 3x3 rotation matrix.
    """
    return fast_rot_mat_p2rtn(nu).T @ fast_rot_mat_eci2rtn(r_eci, v_eci)


# Coordinate transformation instead of Tudat for optimization
@njit(cache=True, fastmath=True)
def fast_cart2kep(cart_state: np.ndarray, mu: float) -> np.ndarray:
    """
    Converts Cartesian state to Keplerian elements.
    Args:
        - cart_state (np.ndarray): Cartesian state vector (6,).
        - mu (float): Gravitational parameter of the central body.
    Returns:
        - np.ndarray: Keplerian elements array [a, e, i, w, W, nu].
    """
    r_vec = cart_state[0:3]
    v_vec = cart_state[3:6]
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    vr = np.dot(r_vec, v_vec) / r

    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    i = np.arccos(max(-1.0, min(1.0, h_vec[2] / h)))

    n_vec = np.array([-h_vec[1], h_vec[0], 0.0])
    n = np.linalg.norm(n_vec)

    if n < 1e-12:
        W = 0.0
        n_vec = np.array([1.0, 0.0, 0.0])
    else:
        W = np.arccos(max(-1.0, min(1.0, n_vec[0] / n)))
        if n_vec[1] < 0:
            W = 2 * np.pi - W

    e_vec = ((v**2 - mu / r) * r_vec - (r * vr) * v_vec) / mu
    e = np.linalg.norm(e_vec)

    if n < 1e-12:
        w = np.arccos(max(-1.0, min(1.0, e_vec[0] / max(e, 1e-12))))
        if e_vec[1] < 0:
            w = 2 * np.pi - w
    elif e > 1e-12:
        w = np.arccos(max(-1.0, min(1.0, np.dot(n_vec, e_vec) / (n * e))))
        if e_vec[2] < 0:
            w = 2 * np.pi - w
    else:
        w = 0.0

    if e > 1e-12:
        nu = np.arccos(max(-1.0, min(1.0, np.dot(e_vec, r_vec) / (e * r))))
        if vr < 0:
            nu = 2 * np.pi - nu
    else:
        if n > 1e-12:
            nu = np.arccos(max(-1.0, min(1.0, np.dot(n_vec, r_vec) / (n * r))))
            if r_vec[2] < 0:
                nu = 2 * np.pi - nu
        else:
            nu = np.arccos(max(-1.0, min(1.0, r_vec[0] / r)))
            if r_vec[1] < 0:
                nu = 2 * np.pi - nu

    a = h**2 / (mu * (1 - e**2))
    return np.array([a, e, i, w, W, nu])


# GVE calculation
@njit(cache=True, fastmath=True)
def fast_gve_matrix_rtn(kep: np.ndarray, mu: float) -> np.ndarray:
    """
    Computes the Gauss Variational Equations (GVE) matrix in the RTN frame.
    Args:
        - kep (np.ndarray): Current Keplerian elements.
        - mu (float): Gravitational parameter of the central body.
    Returns:
        - np.ndarray: 6x3 GVE matrix.
    """
    a, e, i, w, W, nu = kep[0], kep[1], kep[2], kep[3], kep[4], kep[5]

    p = a * (1 - e**2)
    h = np.sqrt(mu * p)
    r = p / (1 + e * np.cos(nu))

    s_nu = np.sin(nu)
    c_nu = np.cos(nu)
    s_u = np.sin(nu + w)
    c_u = np.cos(nu + w)
    s_i = np.sin(i)
    c_i = np.cos(i)

    safe_e = max(1e-6, e)
    safe_si = max(1e-6, s_i)

    # Pre-compute terms
    term_a1 = (2 * a**2 * e * s_nu) / h
    term_a2 = (2 * a**2 * p) / (h * r)

    term_e1 = (p * s_nu) / h
    term_e2 = ((p + r) * c_nu + r * e) / h

    term_i3 = (r * c_u) / h

    term_w1 = (-p * c_nu) / (safe_e * h)
    term_w2 = ((p + r) * s_nu) / (safe_e * h)
    term_w3 = (-r * s_u * c_i) / (h * safe_si)

    term_W3 = (r * s_u) / (h * safe_si)

    term_nu1 = (p * c_nu) / (safe_e * h)
    term_nu2 = -(p + r) * s_nu / (safe_e * h)

    # Construct Matrix manually for speed
    mat = np.zeros((6, 3))

    mat[0, 0] = term_a1
    mat[0, 1] = term_a2

    mat[1, 0] = term_e1
    mat[1, 1] = term_e2

    mat[2, 2] = term_i3

    mat[3, 0] = term_w1
    mat[3, 1] = term_w2
    mat[3, 2] = term_w3

    mat[4, 2] = term_W3

    mat[5, 0] = term_nu1
    mat[5, 1] = term_nu2

    return mat


# Maths (Q and Q_terms) for logging
@njit(cache=True, fastmath=True)
def compute_lyapunov_for_logging(
    kep: np.ndarray,
    dynamic_target: np.ndarray,
    weights: np.ndarray,
    q_law_params: np.ndarray,
    max_rates: np.ndarray,
    is_stage_2: bool,
) -> tuple[float, np.ndarray]:
    """
    Computes the Lyapunov function and its breakdown into terms for logging purposes.
    Args:
         - kep (np.ndarray): Current Keplerian elements.
         - dynamic_target (np.ndarray): Current target Keplerian elements.
         - weights (np.ndarray): Weights for each orbital element in the Q function.
         - q_law_params (np.ndarray): Parameters for the Q-law function.
     returns:
         - Q_total (float): Total Lyapunov function value.
         - q_terms (np.ndarray): Breakdown of Q into individual terms for each element.
    """
    a, e, i, w, W, nu_chaser = kep
    m, n, r = q_law_params[0], q_law_params[1], q_law_params[2]
    k_imp, rp_min, w_imp = q_law_params[3], q_law_params[4], q_law_params[5]
    k_esc, a_max_limit, w_esc = q_law_params[6], q_law_params[7], q_law_params[8]
    W_L, W_scl = q_law_params[9], q_law_params[10]

    # Protect against divide-by-zero
    max_a = max_rates[0] if abs(max_rates[0]) > 1e-12 else 1e-6
    max_e = max_rates[1] if abs(max_rates[1]) > 1e-12 else 1e-6
    max_i = max_rates[2] if abs(max_rates[2]) > 1e-12 else 1e-6
    max_w = max_rates[3] if abs(max_rates[3]) > 1e-12 else 1e-6
    max_W = max_rates[4] if abs(max_rates[4]) > 1e-12 else 1e-6

    # FIX: Use dynamic target for SMA baseline as well
    a_T = dynamic_target[0]

    if is_stage_2:
        # Shortest angular distance wrapped between -pi and pi
        u_chaser = w + nu_chaser
        u_target = dynamic_target[3] + dynamic_target[5]
        delta_u = (u_chaser - u_target + np.pi) % (2.0 * np.pi) - np.pi
        K = (2.0 * W_L / np.pi) * (a_T - rp_min / (1.0 - e))
        a_T = K * np.arctan(W_scl * delta_u) + a_T

    # Calculate Terms (FIX: All errors now calculated against dynamic_target)
    term_a = 0.0
    if weights[0] > 0:
        d_a = a - a_T
        S_a = (1.0 + abs(d_a / (m * a_T)) ** n) ** (1.0 / r)
        term_a = weights[0] * S_a * (d_a / max_a) ** 2

    term_e = (
        weights[1] * ((e - dynamic_target[1]) / max_e) ** 2 if weights[1] > 0 else 0.0
    )
    term_i = (
        weights[2] * ((i - dynamic_target[2]) / max_i) ** 2 if weights[2] > 0 else 0.0
    )

    term_w = 0.0
    if weights[3] > 0:
        d_w = (w - dynamic_target[3] + np.pi) % (2.0 * np.pi) - np.pi
        term_w = weights[3] * (d_w / max_w) ** 2

    term_W = 0.0
    if weights[4] > 0:
        d_W = (W - dynamic_target[4] + np.pi) % (2.0 * np.pi) - np.pi
        term_W = weights[4] * (d_W / max_W) ** 2

    # Penalty Functions
    rp = a * (1.0 - e)
    P_imp = np.exp(k_imp * (1.0 - rp / rp_min)) if rp < rp_min else 0.0
    P_esc = np.exp(k_esc * (a - a_max_limit) / a_max_limit) if a > a_max_limit else 0.0
    penalty = 1.0 + (w_imp * P_imp) + (w_esc * P_esc)

    q_terms = np.array([term_a, term_e, term_i, term_w, term_W]) * penalty
    q_tot = np.sum(q_terms)

    return q_tot, q_terms


# State vector standardization helper
def standardize_state_vector(
    state_definition: "StateDefinition", mu: float, epoch: float
) -> None:
    """
    Converts any state type and frame into J2000 Keplerian elements and updates the object.
    Args:
        - state_definition (StateDefinition): The input state definition object to be updated.
        - mu (float): Gravitational parameter of the central body.
        - epoch (float): Simulation epoch in seconds since J2000.
    Returns:
        - None: The object is modified in place.
    """
    vals = state_definition.values

    # Convert to Cartesian if necessary
    if state_definition.type == CoordinateType.KEPLERIAN:
        cart_vals = element_conversion.keplerian_to_cartesian(vals, mu)
    else:
        cart_vals = vals

    # Handle Frame Rotation to J2000
    if state_definition.frame != ReferenceFrame.J2000:
        R = spice.compute_rotation_matrix_between_frames(
            state_definition.frame.name, "J2000", epoch
        )
        pos = R @ cart_vals[0:3]
        vel = R @ cart_vals[3:6]
        cart_vals = np.concatenate((pos, vel))

    # Update the object with standard values and labels
    state_definition.values = element_conversion.cartesian_to_keplerian(cart_vals, mu)
    state_definition.type = CoordinateType.KEPLERIAN
    state_definition.frame = ReferenceFrame.J2000


# Helpers for Q-Law guidance
@njit(cache=True, fastmath=True)
def wrap_to_pi(angle: float) -> float:
    """
    Wraps an angle to the [-pi, pi] interval.
    Args:
        - angle (float): Input angle in radians.
    Returns:
        - float: Wrapped angle in radians.
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@njit(cache=True, fastmath=True)
def fast_circular_rolling_mean(
    kep_history: np.ndarray, current_osc_state: np.ndarray
) -> np.ndarray:
    """
    Computes the mean Keplerian state from a history array over 1 orbital period.
    Uses linear means for a, e and circular means for angles (i, w, W).
    """
    n = kep_history.shape[0]

    if n < 2:
        return current_osc_state.copy()

    mean_a = np.mean(kep_history[:, 0])
    mean_e = np.mean(kep_history[:, 1])

    mean_i = np.arctan2(
        np.sum(np.sin(kep_history[:, 2])), np.sum(np.cos(kep_history[:, 2]))
    ) % (2 * np.pi)
    mean_w = np.arctan2(
        np.sum(np.sin(kep_history[:, 3])), np.sum(np.cos(kep_history[:, 3]))
    ) % (2 * np.pi)
    mean_W = np.arctan2(
        np.sum(np.sin(kep_history[:, 4])), np.sum(np.cos(kep_history[:, 4]))
    ) % (2 * np.pi)

    osc_w = current_osc_state[3]
    osc_nu = current_osc_state[5]
    u_osc = (osc_w + osc_nu) % (2 * np.pi)
    nu_eff = (u_osc - mean_w) % (2 * np.pi)

    return np.array([mean_a, mean_e, mean_i, mean_w, mean_W, nu_eff])
