from guidance.base_q_law import BaseQLaw
from guidance.math_utils import shared_math, sail_math
from config import SailForceModel, SimulationConfig, EclipseModel

from tudatpy.astro import fundamentals

import numpy as np


class SailQLaw(BaseQLaw):
    """
    Q-Law guidance logic specific to solar sail propulsion.

    Attributes:
        - area_to_mass (float): The sail's area-to-mass ratio in m^2/kg.
        - initial_state_kep (np.ndarray): The starting Keplerian elements of the spacecraft.
        - target_state_kep (np.ndarray): The target Keplerian elements.
        - mu_sun (float): Sun's gravitational parameter.
        - model_id (int): Integer identifier for the active sail force model (IDEAL or others).
        - last_aspect_angle (float): Cached angle between the Sun vector and the orbit normal.
    """

    def __init__(self, bodies: dict, sim_config: "SimulationConfig") -> None:
        """
        Initializes the Q-Law guidance logic specific to solar sail propulsion.
        Args:
            - bodies (dict): Tudat system of bodies.
            - sim_config (SimulationConfig): The active simulation configuration.
        Returns:
            - None
        """
        super().__init__(bodies, sim_config)

        self.area_to_mass = self.param.config.spacecraft.area_to_mass
        self.initial_state_kep = self.param.config.initial_state.values
        self.target_state_kep = self.param.config.target_state.values
        self.mu_sun = self.param.mu_sun

        self.model_id = self.param.config.spacecraft.model

        model_enum = self.param.config.spacecraft.model
        if model_enum == SailForceModel.IDEAL:
            self.model_id = sail_math.IDEAL
            self.model_params = np.zeros(3)
        else:
            raise ValueError(f"Unsupported sail force model: {model_enum}")

        self.last_aspect_angle = 0.0
        self.last_shadow_factor = 1.0  # Default to full sun

        # Extract Sun radius for the eclipse model
        if self.param.config.eclipse_model == EclipseModel.CONICAL:
            self.sun_radius = bodies.get("Sun").shape_model.average_radius

        max_rates = sail_math.compute_max_rates_sail_triple_min(
            self.P_1AU, self.area_to_mass, self.initial_state_kep, self.mu_earth
        )
        self.var.max_rates = np.array(max_rates)

    def get_accel_vec_eci(self, time: float) -> np.ndarray:
        """
        Calculates the required ECI acceleration vector at a given simulation time.
        Args:
            - time (float): Current simulation epoch in seconds.
        Returns:
            - np.ndarray: Acceleration vector in the ECI frame (3,).
        """
        self.update_state(time)
        # Get States
        state_sun = self.bodies.get("Sun").ephemeris.cartesian_state(time)
        state_earth = self.bodies.get("Earth").ephemeris.cartesian_state(time)
        state_scpacecraft = self.bodies.get("Spacecraft").state
        r_sc_eci = state_scpacecraft[0:3]
        v_sc_eci = state_scpacecraft[3:6]

        # Compute important vectors
        s_vec_eci = r_sc_eci - (state_sun[0:3] - state_earth[0:3])
        dist_sun = np.linalg.norm(s_vec_eci)
        s_vec_eci = s_vec_eci / dist_sun 
        self.var.s_vec_eci = s_vec_eci

        h_vec = np.cross(r_sc_eci, v_sc_eci)
        h_norm = h_vec / np.linalg.norm(h_vec)

        # Rotation matrices
        R_eci2s = shared_math.fast_rot_mat_eci2s(s_vec_eci, self.R_eci2ecl)
        R_s2rtn = shared_math.fast_rot_mat_eci2rtn(r_sc_eci, v_sc_eci) @ R_eci2s.T

        # Aspect angle calc
        dot_prod = np.dot(h_norm, s_vec_eci)
        theta_h_s = np.arccos(np.clip(dot_prod, -1.0, 1.0))
        self.last_aspect_angle = min(theta_h_s, np.pi - theta_h_s)

        # Staging strategy
        if self.var.current_stage == 1 or self.var.current_stage == 2:
            kep_sc_state = self.var.chaser_mean_kep
            kep_tgt_state = self.var.target_mean_kep
        else:
            kep_sc_state = self.var.chaser_osc_kep
            kep_tgt_state = self.var.target_osc_kep

        is_phasing_active = self.var.current_stage in [0, 2, 3]

        # Math Calls
        grad = sail_math.fast_q_gradient_sail(
            kep_sc_state,
            kep_tgt_state,
            self.weights,
            self.q_law_math_params,
            self.var.max_rates,
            is_phasing_active,
        )
        G = shared_math.fast_gve_matrix_rtn(self.var.chaser_osc_kep, self.mu_earth)

        # C. Primer Vector & Target direction
        primer_l = -G[:5, :].T @ grad[:5].T
        primer_s = R_s2rtn.T @ primer_l

        alpha_star, beta_star = sail_math.compute_optimal_angles_sail(
            primer_s, self.model_id, self.model_params
        )
        self.p_alpha = alpha_star
        self.p_beta = beta_star

        a_s = sail_math.get_acceleration_vector_s(
            alpha_star,
            beta_star,
            dist_sun,
            self.area_to_mass,
            self.model_id,
            self.P_1AU,
            self.AU,
        )

        if self.param.config.eclipse_model == EclipseModel.CONICAL:
            earth_centered_sun_pos = state_sun[0:3] - state_earth[0:3]
            earth_centered_earth_pos = np.zeros(3)
            shadow_factor = fundamentals.compute_shadow_function(
                occulted_body_position=earth_centered_sun_pos,
                occulted_body_radius=self.sun_radius,
                occulting_body_position=earth_centered_earth_pos,
                occulting_body_radius=self.earth_radius,
                satellite_position=r_sc_eci,
            )
        else:
            shadow_factor = 1.0
        self.last_shadow_factor = float(shadow_factor)

        acc_vec_eci = R_eci2s.T @ (a_s * shadow_factor)

        # LOGGING
        if self.param.config.logging.log_step_size > 0:
            if (time - self.last_print_time) >= self.param.config.logging.log_step_size:
                self.daily_output()

        return acc_vec_eci
 