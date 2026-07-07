from config import (
    SimulationParameters,
    SimulationVariables,
    SimulationConfig,
)
from guidance.math_utils import shared_math

import logging
import numpy as np
from collections import deque

from tudatpy.interface import spice


class BaseQLaw:
    """
    Abstract base class for Q-Law guidance strategies. Manages shared state,
    caching, and environment variables across different propulsion types.

    Attributes:
        - param (SimulationParameters): Fixed simulation parameters and constants.
        - var (SimulationVariables): Dynamically updating simulation states.
        - bodies (dict): Tudat environment body objects.
        - target_state (np.ndarray): The initial target Keplerian elements (6,).
        - weights (np.ndarray): Penalty weights for Q-Law targeting (6,).
        - q_law_math_params (np.ndarray): Array of scaling and penalty tuning parameters.
        - mu_earth (float): Earth's gravitational parameter.
        - earth_radius (float): Earth's equatorial radius.
        - AU (float): Astronomical Unit in meters.
        - P_1AU (float): Solar radiation pressure at 1 AU.
        - sim_start_epoch (float): Mission start time in seconds since J2000.
        - R_eci2ecl (np.ndarray): Static rotation matrix from J2000 to ECLIPJ2000 at start epoch.
        - mean_motion_target (float): Constant mean motion of the target orbit.
        - last_print_time (float): Epoch of the last terminal logging output.
        - p_alpha (float): Cached pitch control angle in radians.
        - p_beta (float): Cached clock control angle in radians.
        - _cached_time (float): The simulation epoch of the last acceleration calculation.
        - _cached_acc_vec (np.ndarray): Cached acceleration vector to avoid recomputation.
        - _cached_thrust_mag (float): Cached thrust magnitude.
        - stage_2_start_time (float): Epoch when rendezvous stage 2 was initiated.
    """

    def __init__(self, bodies: dict, sim_config: "SimulationConfig") -> None:
        """
        Initializes the abstract base class for Q-Law guidance strategies.
        Args:
            - bodies (dict): Tudat system of bodies.
            - sim_config (SimulationConfig): The active simulation configuration.
        Returns:
            - None
        """
        self.param = SimulationParameters(config=sim_config)
        self.var = SimulationVariables()
        self.bodies = bodies

        self.target_state = self.param.config.target_state.values # This represent the initial position of the target

        # Q-Law parameters
        g = self.param.config.guidance
        self.weights = g.weights
        self.q_law_math_params = np.array(
            [
                g.m,
                g.n,
                g.r,
                g.k_imp,
                g.rp_min,
                g.w_imp,
                g.k_esc,
                g.a_max_limit,
                g.w_esc,
                g.W_L,
                g.W_scl,
            ]
        )

        # Constants
        self.mu_earth = self.param.mu_earth
        self.earth_radius = self.param.earth_radius
        self.AU = self.param.AU
        self.P_1AU = self.param.P_1AU

        # Others
        self.sim_start_epoch = self.param.config.sim_start_epoch
        self.R_eci2ecl = spice.compute_rotation_matrix_between_frames(
            "J2000", "ECLIPJ2000", self.sim_start_epoch
        )
        self.last_print_time = -np.inf

        # Variables for plotting control angles
        self.p_alpha = 0.0
        self.p_beta = 0.0

        # Caching variables to avoid re-computing acceleration multiple times per step
        self._cached_time = -1.0
        self._cached_acc_vec = np.zeros(3)
        self._cached_thrust_mag = 0.0

        # Rendezvous staging
        self.active_stages_for_simulation = self.param.config.stages
        self.current_stage_idx = 0
        self.current_stage = (
            self.active_stages_for_simulation[self.current_stage_idx]
            if self.active_stages_for_simulation
            else 0
        )
        self.var.current_stage = self.current_stage
        self.transition_log = []

        # Mean element calculation variables
        self.history_time = deque()
        self.history_kep_chaser_osc = deque()
        self.history_kep_target_osc = deque()

    def update_state(self, time: float) -> None:
        """
        Updates internal variables with the current spacecraft and target states,
        and calculates the rolling mean elements to filter out J2 noise.
        """
        self.var.time = time
        self.var.mass = self.bodies.get("Spacecraft").mass

        self.var.chaser_osc_kep = shared_math.fast_cart2kep(
            self.bodies.get("Spacecraft").state, self.mu_earth
        )
        self.var.target_osc_kep = shared_math.fast_cart2kep(
            self.bodies.get("Target").state, self.mu_earth
        )

        # --- MEAN ELEMENT FILTERING ---
        self.history_time.append(time)
        self.history_kep_chaser_osc.append(self.var.chaser_osc_kep)
        self.history_kep_target_osc.append(self.var.target_osc_kep)

        # Prune history older than one orbital period
        current_period = (
            2 * np.pi * np.sqrt((self.var.chaser_osc_kep[0] ** 3) / self.mu_earth)
        )
        while self.history_time and (time - self.history_time[0]) > current_period:
            self.history_time.popleft()
            self.history_kep_chaser_osc.popleft()
            self.history_kep_target_osc.popleft()

        # Compute and store the mathematically stable mean states
        self.var.chaser_mean_kep = shared_math.fast_circular_rolling_mean(
            np.array(self.history_kep_chaser_osc), self.var.chaser_osc_kep
        )

        self.var.target_mean_kep = shared_math.fast_circular_rolling_mean(
            np.array(self.history_kep_target_osc), self.var.target_osc_kep
        )
        # --------------------------------

        # Staging
        self.evaluate_stage_transitions(time)
        self.current_stage = self.active_stages_for_simulation[self.current_stage_idx]
        self.var.current_stage = self.current_stage

    def daily_output(self) -> None:
        """
        Prints a snapshot of current simulation progress and Lyapunov terms to the terminal.
        Args:
            - None
        Returns:
            - None
        """
        if self.current_stage in [1, 2]:
            self.var.kep_sc_state = self.var.chaser_mean_kep
            self.var.kep_tgt_state = self.var.target_mean_kep
        else:
            self.var.kep_sc_state = self.var.chaser_osc_kep
            self.var.kep_tgt_state = self.var.target_osc_kep

        a, e, i, w, W, nu = self.var.kep_sc_state
        target_w, target_nu = (
            self.var.kep_tgt_state[3],
            self.var.kep_tgt_state[5],
        )

        days = (self.var.time - self.sim_start_epoch) / 86400.0
        max_rates = self.var.max_rates

        # Calculate Q terms for display
        is_stage_2 = getattr(self.var, "current_stage", 1) == 2
        Q_total, q_terms = shared_math.compute_lyapunov_for_logging(
            self.var.kep_sc_state,
            self.var.kep_tgt_state,
            self.weights,
            self.q_law_math_params,
            max_rates,
            is_stage_2,
        )

        logging.info(f"\n--- Day {days:.1f} | STAGE {self.current_stage} ---")
        logging.info(
            f"State: a={a/1e3:.1f}km, e={e:.4f}, i={np.rad2deg(i):.2f}deg, "
            f"w={np.rad2deg(w):.2f}deg, RAAN={np.rad2deg(W):.2f}deg, "
            f"u={np.rad2deg(shared_math.wrap_to_pi(w+nu)):.2f}deg (Target u={np.rad2deg(shared_math.wrap_to_pi(target_nu + target_w)):.2f}deg)"
        )
        logging.info(f"Q_tot: {Q_total:.2e}")
        logging.info(
            f"Breakdown -> Qa: {q_terms[0]:.2e} | Qe: {q_terms[1]:.2e} | "
            f"Qi: {q_terms[2]:.2e} | Qw: {q_terms[3]:.2e} | QW: {q_terms[4]:.2e}"
        )
        self.last_print_time = self.var.time

    def get_convergence_error(self) -> float:
        """
        Calculates the specific error metrics required for the currently active stage.
        Used by both the internal state machine and the propagator termination logic.
        """
        stage = getattr(self, "current_stage", 0)
        weights = self.weights
        errors = []

        if stage == 3 or stage == 0:
            sc_cart = self.bodies.get("Spacecraft").state
            tgt_cart = self.bodies.get("Target").state

            rel_pos = sc_cart[:3] - tgt_cart[:3]
            dist_error = (
                np.linalg.norm(rel_pos) / 1000.0
            ) / self.param.config.stage_3_tol_distance

            rel_vel = sc_cart[3:] - tgt_cart[3:]
            vel_error = np.linalg.norm(rel_vel) / self.param.config.stage_3_tol_velocity

            return max(dist_error, vel_error)

        elif stage == 1:
            sc_keplerian_mean = self.var.chaser_mean_kep
            tgt_keplerian_mean = self.var.target_mean_kep
            tol = self.param.config.stage_1_tol
            tol_ang = self.param.config.stage_1_tol_angle

            if weights[0] > 0:
                errors.append(
                    abs(
                        (sc_keplerian_mean[0] - tgt_keplerian_mean[0])
                        / tgt_keplerian_mean[0]
                    )
                    / tol
                )
            if weights[1] > 0:
                errors.append(abs(sc_keplerian_mean[1] - tgt_keplerian_mean[1]) / tol)
            if weights[2] > 0:
                errors.append(
                    abs(sc_keplerian_mean[2] - tgt_keplerian_mean[2]) / tol_ang
                )
            if weights[3] > 0:
                dw = abs(sc_keplerian_mean[3] - tgt_keplerian_mean[3])
                errors.append(min(dw, 2 * np.pi - dw) / tol_ang)
            if weights[4] > 0:
                dW = abs(sc_keplerian_mean[4] - tgt_keplerian_mean[4])
                errors.append(min(dW, 2 * np.pi - dW) / tol_ang)

            return max(errors)

        elif stage == 2:
            sc_keplerian_mean = self.var.chaser_mean_kep
            tgt_keplerian_mean = self.var.target_mean_kep
            tol = self.param.config.stage_2_tol
            tol_ang = self.param.config.stage_2_tol_angle

            u_sc = sc_keplerian_mean[3] + sc_keplerian_mean[5]
            u_tgt = tgt_keplerian_mean[3] + tgt_keplerian_mean[5]
            du = abs((u_sc - u_tgt + np.pi) % (2 * np.pi) - np.pi)
            errors.append(du / tol_ang)

            if weights[0] > 0:
                errors.append(
                    abs(
                        (sc_keplerian_mean[0] - tgt_keplerian_mean[0])
                        / tgt_keplerian_mean[0]
                    )
                    / tol
                )
            if weights[1] > 0:
                errors.append(abs(sc_keplerian_mean[1] - tgt_keplerian_mean[1]) / tol)
            if weights[2] > 0:
                errors.append(
                    abs(sc_keplerian_mean[2] - tgt_keplerian_mean[2]) / tol_ang
                )
            if weights[3] > 0:
                dw = abs(sc_keplerian_mean[3] - tgt_keplerian_mean[3])
                errors.append(min(dw, 2 * np.pi - dw) / tol_ang)
            if weights[4] > 0:
                dW = abs(sc_keplerian_mean[4] - tgt_keplerian_mean[4])
                errors.append(min(dW, 2 * np.pi - dW) / tol_ang)

            return max(errors)

    def evaluate_stage_transitions(self, current_time: float) -> None:
        """Native state machine to evaluate if we should advance to the next stage."""
        errors = self.get_convergence_error()
        if self.current_stage_idx >= len(self.active_stages_for_simulation) - 1:
            return

        next_stage = self.active_stages_for_simulation[self.current_stage_idx + 1]

        # Transition 1 -> 2 
        if self.current_stage == 1 and next_stage == 2:
            if errors < 1.0:
                self._advance_stage(current_time, 2)
                return

        # Transition 2 -> 3 
        elif self.current_stage == 2 and next_stage == 3:
            if errors < 1.0:
                self._advance_stage(current_time, 3)
                return

    def _advance_stage(self, current_time, new_stage):
        self.current_stage_idx += 1
        self.current_stage = new_stage
        self.transition_log.append((current_time, new_stage))
        print(
            f"\n[TIME: {current_time/86400:.2f} days] --- TRANSITIONING TO STAGE {new_stage} ---"
        )

    def get_detailed_errors(self) -> np.ndarray:
        """Returns an array of specific normalized errors: [err_a, err_e, err_u]"""
        if self.current_stage not in [1, 2]:
            return np.zeros(3)

        sc_mean = self.var.chaser_mean_kep
        tgt_mean = self.var.target_mean_kep

        # Dynamically fetch the tolerances for the current stage
        tol_a = (
            self.param.config.stage_1_tol
            if self.current_stage == 1
            else self.param.config.stage_2_tol
        )
        tol_e = (
            self.param.config.stage_1_tol
            if self.current_stage == 1
            else self.param.config.stage_2_tol
        )
        tol_ang = (
            self.param.config.stage_1_tol_angle
            if self.current_stage == 1
            else self.param.config.stage_2_tol_angle
        )

        # 1. SMA Normalized Error
        err_a = abs((sc_mean[0] - tgt_mean[0]) / tgt_mean[0]) / tol_a

        # 2. Eccentricity Normalized Error
        err_e = abs(sc_mean[1] - tgt_mean[1]) / tol_e

        # 3. Argument of Latitude (Phase) Normalized Error
        u_sc = sc_mean[3] + sc_mean[5]
        u_tgt = tgt_mean[3] + tgt_mean[5]
        du = abs((u_sc - u_tgt + np.pi) % (2 * np.pi) - np.pi)
        err_u = du / tol_ang

        return np.array([err_a, err_e, err_u])
