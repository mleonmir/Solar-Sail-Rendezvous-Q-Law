from guidance.base_q_law import BaseQLaw
from guidance.math_utils import ion_math, shared_math
from config import SimulationConfig


from tudatpy.astro import element_conversion

import numpy as np


class IonQLaw(BaseQLaw):
    """
    Q-Law guidance logic specific to continuous, fixed-thrust ion propulsion.

    Attributes:
        - thrust (float): Spacecraft engine thrust magnitude in Newtons.
        - isp (float): Spacecraft engine specific impulse in seconds.
        - weights (np.ndarray): Penalty weights for Q-Law targeting.
    """ 

    def __init__(self, bodies: dict, sim_config: "SimulationConfig") -> None:
        """
        Initializes the Q-Law guidance logic specific to continuous ion propulsion.
        Args:
            - bodies (dict): Tudat system of bodies.
            - sim_config (SimulationConfig): The active simulation configuration.
        Returns:
            - None
        """
        super().__init__(bodies, sim_config)

        self.thrust = self.param.config.spacecraft.thrust
        self.isp = self.param.config.spacecraft.isp
        self.weights = self.param.config.guidance.weights

    def get_accel_vec_eci(self, time: float) -> np.ndarray:
        """
        Calculates the required ECI acceleration vector for the spacecraft at a given simulation epoch using Q-Law guidance.
        Args:
            - time (float): Current simulation epoch in seconds.
        Returns:
            - np.ndarray: Acceleration vector in the ECI frame (3,).
        """
        # --- General Variables Initialization ---
        if time == self._cached_time:
            return self._cached_acc_vec

        self.update_state(time)
        mass = self.var.mass

        # --- Stage Logic for Element Selection ---
        # (The stage transitions are already handled automatically by base_q_law.py)
        if self.var.current_stage in [1, 2]:
            kep_sc_state = self.var.chaser_mean_kep
            kep_tgt_state = self.var.target_mean_kep
        else:
            kep_sc_state = self.var.chaser_osc_kep
            kep_tgt_state = self.var.target_osc_kep

        is_stage_2 = self.var.current_stage == 2

        # --- Primer calculation (gradient, max_rates, GVE) ---
        max_rates = ion_math.compute_max_rates_ion(
            kep_sc_state, self.thrust, mass, self.param.mu_earth
        )
        self.var.max_rates = max_rates

        grad = ion_math.fast_q_gradient_ion(
            kep_sc_state,
            kep_tgt_state,
            self.weights,
            self.q_law_math_params,
            max_rates,
            is_stage_2,
        )

        G = shared_math.fast_gve_matrix_rtn(
            self.var.chaser_osc_kep, self.param.mu_earth
        )

        primer_rtn = G[:5, :].T @ grad[:5]

        # --- Acceleration vector computation ---
        norm_primer = np.linalg.norm(primer_rtn)

        if norm_primer > 1e-15:
            u_rtn = -primer_rtn / norm_primer
            self.p_alpha = np.arcsin(np.clip(u_rtn[2], -1.0, 1.0))
            self.p_beta = np.arctan2(u_rtn[1], u_rtn[0])
            cart_state = element_conversion.keplerian_to_cartesian(
                self.var.chaser_osc_kep, self.param.mu_earth
            )
            R_rtn2eci = shared_math.fast_rot_mat_eci2rtn(
                cart_state[0:3], cart_state[3:6]
            ).T
            u_eci = R_rtn2eci @ u_rtn
            acc_vec_eci = u_eci * (self.thrust / mass)
        else:
            acc_vec_eci = np.zeros(3)

        # -- Caching for time saving if potential re-use ---
        self._cached_time = time
        self._cached_acc_vec = acc_vec_eci
        self._cached_thrust_mag = self.thrust

        # -- Logging ---
        if self.param.config.logging.log_step_size > 0:
            if (time - self.last_print_time) >= self.param.config.logging.log_step_size:
                self.daily_output()

        return acc_vec_eci

    # Helper function for Tudat (see propagator)
    def get_thrust_magnitude(self, time: float) -> float:
        """
        Returns the instantaneous thrust magnitude, checking the cache first.
        Args:
            - time (float): Current simulation epoch in seconds.
        Returns:
            - float: Commanded thrust in Newtons (or 0.0 if coasting).
        """
        if time != self._cached_time:
            self.get_accel_vec_eci(time)
        return self._cached_thrust_mag
