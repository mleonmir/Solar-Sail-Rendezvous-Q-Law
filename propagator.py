import numpy as np

import logging

from tudatpy import dynamics
from tudatpy.dynamics import propagation_setup, environment_setup
from tudatpy.util import result2array
from tudatpy.astro import element_conversion

from config import (
    PropulsionType,
    IntegratorType,
    GravityModel,
    SimulationConfig,
    SpacecraftConfig,
)
from guidance.q_law_ion import IonQLaw
from guidance.q_law_sail import SailQLaw
from guidance.math_utils import shared_math
from guidance.base_q_law import BaseQLaw


def get_normalized_target_error(guidance: "BaseQLaw") -> np.ndarray:
    """
    Intreface between the propagator and the guidance logic to compute the normalized error for termination checking.
    """
    final_stage = (
        guidance.param.config.stages[-1] if guidance.param.config.stages else 0
    )

    if guidance.var.current_stage == final_stage:
        return np.array([guidance.get_convergence_error()])
    else:
        return np.array([2])


def create_environment(sim_config: SimulationConfig) -> dict:
    """
    Initializes the Tudat system of bodies for the simulation.
    Args:
        - sim_config (SimulationConfig): The active simulation configuration.
    Returns:
        - dict: Tudat system of bodies object.
    """
    bodies_to_create = (
        ["Earth", "Sun"]
        if sim_config.spacecraft.mode == PropulsionType.SAIL
        else ["Earth"]
    )
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, "Earth", "J2000"
    )

    # Chaser
    body_settings.add_empty_settings("Spacecraft")
    body_settings.get("Spacecraft").constant_mass = sim_config.spacecraft.initial_mass

    # Target
    body_settings.add_empty_settings("Target")
    body_settings.get("Target").constant_mass = 1000.0  # Arbitrary mass, unpowered

    return environment_setup.create_system_of_bodies(body_settings)


def create_acceleration_models(
    bodies: dict, sim_config: "SimulationConfig", guidance: "BaseQLaw"
) -> dict:
    """
    Creates the acceleration models for the spacecraft and environment.
    Args:
        - bodies (dict): Tudat system of bodies.
        - sim_config (SimulationConfig): The active simulation configuration.
        - guidance (BaseQLaw): The active guidance logic instance.
    Returns:
        - dict: Tudat acceleration models dictionary.
    """
    if sim_config.earth_gravity_model == GravityModel.POINT_MASS:
        earth_accel = propagation_setup.acceleration.point_mass_gravity()
    elif sim_config.earth_gravity_model == GravityModel.J2:
        earth_accel = propagation_setup.acceleration.spherical_harmonic_gravity(2, 0)
    else:
        earth_accel = propagation_setup.acceleration.point_mass_gravity()

    # Chaser Accelerations
    accelerations_settings_sc = {"Earth": [earth_accel]}
    if sim_config.spacecraft.mode == PropulsionType.SAIL:
        accelerations_settings_sc["Sun"] = [
            propagation_setup.acceleration.custom_acceleration(
                guidance.get_accel_vec_eci
            )
        ]
    elif sim_config.spacecraft.mode == PropulsionType.ION:
        accelerations_settings_sc["Spacecraft"] = [
            propagation_setup.acceleration.custom_acceleration(
                guidance.get_accel_vec_eci
            )
        ]

    # Target Accelerations
    accelerations_settings_target = {"Earth": [earth_accel]}

    acceleration_dict = {
        "Spacecraft": accelerations_settings_sc,
        "Target": accelerations_settings_target,
    }
    bodies_to_propagate = ["Spacecraft", "Target"]
    central_bodies = ["Earth", "Earth"]

    return propagation_setup.create_acceleration_models(
        bodies, acceleration_dict, bodies_to_propagate, central_bodies
    )


def create_mass_rate_models(
    bodies: dict,
    sim_config: "SimulationConfig",
    guidance: "BaseQLaw",
    acceleration_models: dict,
) -> dict:
    """
    Creates the mass rate models to simulate fuel depletion for ion engines.
    Args:
        - bodies (dict): Tudat system of bodies.
        - sim_config (SimulationConfig): The active simulation configuration.
        - guidance (BaseQLaw): The active guidance logic instance.
        - acceleration_models (dict): The previously created acceleration models.
    Returns:
        - dict: Tudat mass rate models dictionary.
    """
    mass_rate_models = dict()

    if sim_config.spacecraft.mode == PropulsionType.ION:

        def custom_mass_rate(time):
            if not sim_config.spacecraft.mass_depletion:
                return 0.0
            else:
                thrust_force = guidance.get_thrust_magnitude(time)
                return -thrust_force / (sim_config.spacecraft.isp * guidance.param.g0)

        mass_rate_models["Spacecraft"] = [
            propagation_setup.mass_rate.custom_mass_rate(custom_mass_rate)
        ]

    return propagation_setup.create_mass_rate_models(
        bodies, mass_rate_models, acceleration_models
    )


def get_dependent_variable_save_settings(
    sc_config: "SpacecraftConfig", guidance: "BaseQLaw"
) -> list:
    """
    Compiles the universal list of dependent variables to track and save during propagation.
    Dynamically routes Q-term logs based on the active stage elements.
    """
    acc_type = (
        propagation_setup.acceleration.AvailableAcceleration.custom_acceleration_type
    )
    target = "Spacecraft" if sc_config.mode.name == "ION" else "Sun"

    def get_q_data():
        """
        Computes the Lyapunov function and its individual terms for logging purposes.
        """
        is_phasing = getattr(guidance.var, "current_stage", 1) in [0, 2, 3]
        if guidance.var.current_stage == 1 or guidance.var.current_stage == 2:
            kep_sc_state = guidance.var.chaser_mean_kep
            kep_tgt_state = guidance.var.target_mean_kep
        else:
            kep_sc_state = guidance.var.chaser_osc_kep
            kep_tgt_state = guidance.var.target_osc_kep

        q_tot, q_terms = shared_math.compute_lyapunov_for_logging(
            kep_sc_state,  # The router dynamically populates this with Mean or Osc
            kep_tgt_state,  # The router dynamically populates this with Mean or Osc
            guidance.weights,
            guidance.q_law_math_params,
            guidance.var.max_rates,
            is_phasing,
        )
        return np.concatenate(([q_tot], q_terms))

    deps = [
        # Cols 1-6: Keplerian State of S/C in ECI (Osculating by default in Tudat)
        propagation_setup.dependent_variable.keplerian_state("Spacecraft", "Earth"),
        # Col 7: Mass
        propagation_setup.dependent_variable.body_mass("Spacecraft"),
        # Col 8: Acceleration Norm
        propagation_setup.dependent_variable.single_acceleration_norm(
            acc_type, "Spacecraft", target
        ),
        # Cols 9-10: Control Angles (Alpha, Beta)
        propagation_setup.dependent_variable.custom_dependent_variable(
            lambda: np.array([guidance.p_alpha, guidance.p_beta]), 2
        ),
        # Col 11: Aspect Angle
        propagation_setup.dependent_variable.custom_dependent_variable(
            lambda: np.array([getattr(guidance, "last_aspect_angle", 0.0)]), 1
        ),
        # Cols 12-17: Q-Values (Total, a, e, i, w, W) -> Routed to active math
        propagation_setup.dependent_variable.custom_dependent_variable(get_q_data, 6),
        # Cols 18-23: Target Keplerian State (Explicitly Osculating)
        propagation_setup.dependent_variable.custom_dependent_variable(
            lambda: guidance.var.target_osc_kep, 6
        ),
        # Col 24: Stage History
        propagation_setup.dependent_variable.custom_dependent_variable(
            lambda: np.array([guidance.var.current_stage]), 1
        ),
        # Cols 25-30: Mean Keplerian State Chaser
        propagation_setup.dependent_variable.custom_dependent_variable(
            lambda: guidance.var.chaser_mean_kep, 6
        ),
        # Cols 31-36: Mean Keplerian State Target
        propagation_setup.dependent_variable.custom_dependent_variable(
            lambda: guidance.var.target_mean_kep, 6
        ),
        # Col 37: Shadow Factor
        propagation_setup.dependent_variable.custom_dependent_variable(
            lambda: np.array([getattr(guidance, "last_shadow_factor", 1.0)]), 1
        ),
        # Cols 38-40: Detailed Normalized Errors [err_a, err_e, err_u]
        propagation_setup.dependent_variable.custom_dependent_variable(
            lambda: guidance.get_detailed_errors(), 3
        ),
    ]
    return deps


def get_integrator_settings(
    sim_config: "SimulationConfig", guidance: "BaseQLaw"
) -> object:
    """
    Creates the settings for the numerical integrator based on the simulation config.
    Args:
        - sim_config (SimulationConfig): The active simulation configuration.
        - guidance (BaseQLaw): The active guidance logic instance.
    Returns:
        - object: Tudat integrator settings object.
    """
    int_cfg = sim_config.integrator

    if int_cfg.type == IntegratorType.RK4:
        integrator_settings = propagation_setup.integrator.runge_kutta_4(
            guidance.sim_start_epoch, int_cfg.initial_step_size
        )
    elif int_cfg.type == IntegratorType.RKF78:
        integrator_settings = (
            propagation_setup.integrator.runge_kutta_variable_step_size(
                initial_time_step=int_cfg.initial_step_size,
                coefficient_set=propagation_setup.integrator.rkf_78,
                minimum_step_size=int_cfg.min_step_size,
                maximum_step_size=int_cfg.max_step_size,
                relative_error_tolerance=int_cfg.rel_tol,
                absolute_error_tolerance=int_cfg.abs_tol,
            )
        )

    return integrator_settings


def print_termination_outputs(
    sim_duration_days: float,
    sim_config: "SimulationConfig",
    state_array: np.ndarray,
    guidance: "BaseQLaw",
) -> None:
    """
    Prints the final state, errors, and termination reason to the console.
    """
    
    final_mass = state_array[-1, 13]
    if sim_duration_days >= sim_config.sim_max_days - 1e-4:
        logging.warning(
            f"Target conditions were NOT reached. Simulation stopped at {sim_duration_days:.3f} days (Maximum time limit)."
        )
    elif final_mass <= 1.001:
        logging.info(
            f"Simulation stopped at {sim_duration_days:.3f} days because spacecraft mass reached {final_mass:.2f} kg."
        )
    else:
        logging.info(
            f"Target conditions successfully reached! Simulation stopped at {sim_duration_days:.3f} days."
        )

    # =========================================================================
    # 1. ORBITAL ELEMENT EXTRACTION
    # =========================================================================
    final_cart_state = state_array[-1, 1:7]
    true_target_cart = state_array[-1, 7:13]

    # Extract true INSTANTANEOUS (Osculating) states directly from the final numerical step
    final_sc_kep_osc = shared_math.fast_cart2kep(
        final_cart_state, guidance.param.mu_earth
    )
    final_target_kep_osc = shared_math.fast_cart2kep(
        true_target_cart, guidance.param.mu_earth
    )

    # Extract Mean states explicitly from our new tracking variables
    final_sc_kep_mean = guidance.var.chaser_mean_kep
    final_target_kep_mean = guidance.var.target_mean_kep

    u_chaser_osc = final_sc_kep_osc[3] + final_sc_kep_osc[5]
    u_target_osc = final_target_kep_osc[3] + final_target_kep_osc[5]

    u_chaser_mean = final_sc_kep_mean[3] + final_sc_kep_mean[5]
    u_target_mean = final_target_kep_mean[3] + final_target_kep_mean[5]


    # Output the infos 
    logging.info("\n--- FINAL STATES AT TERMINATION (MEAN ELEMENTS) ---")
    logging.info(
        f"Spacecraft : a={final_sc_kep_mean[0]/1e3:.3f} km, e={final_sc_kep_mean[1]:.6f}, "
        f"i={np.rad2deg(final_sc_kep_mean[2]):.4f}°, w={np.rad2deg(final_sc_kep_mean[3]):.4f}°, "
        f"W={np.rad2deg(final_sc_kep_mean[4]):.4f}°, nu={np.rad2deg(final_sc_kep_mean[5]):.4f}°"
    )
    logging.info(
        f"Target     : a={final_target_kep_mean[0]/1e3:.3f} km, e={final_target_kep_mean[1]:.6f}, "
        f"i={np.rad2deg(final_target_kep_mean[2]):.4f}°, w={np.rad2deg(final_target_kep_mean[3]):.4f}°, "
        f"W={np.rad2deg(final_target_kep_mean[4]):.4f}°, nu={np.rad2deg(final_target_kep_mean[5]):.4f}°"
    )
    logging.info(f"\n--- FINAL STATES AT TERMINATION (OSCULATING ELEMENTS) ---")
    logging.info(f"chaser_osc = np.array({list(final_sc_kep_osc)})")
    logging.info(f"target_osc = np.array({list(final_target_kep_osc)})")

    # =========================================================================
    # 2. PHYSICAL PROXIMITY ERRORS (OSCULATING)
    # =========================================================================
    rel_vel_mag = np.linalg.norm(final_cart_state[3:6] - true_target_cart[3:6])
    rel_dist_mag = np.linalg.norm(final_cart_state[:3] - true_target_cart[:3])

    logging.info(f"\n--- FINAL RELATIVE STATE (OSCULATING) ---")
    logging.info(f"Relative Velocity: {rel_vel_mag:.3f} m/s")
    logging.info(f"Relative Distance: {rel_dist_mag/1e3:.3f} km\n")

    # =========================================================================
    # 3. ABSOLUTE ERRORS TABLE (MEAN vs OSCULATING)
    # =========================================================================
    def ang_err(c, t):
        return np.rad2deg(abs((c - t + np.pi) % (2 * np.pi) - np.pi))

    # Calculate Mean Errors
    da_m = abs(final_sc_kep_mean[0] - final_target_kep_mean[0]) / 1e3
    de_m = abs(final_sc_kep_mean[1] - final_target_kep_mean[1])
    di_m = np.rad2deg(abs(final_sc_kep_mean[2] - final_target_kep_mean[2]))
    dw_m = ang_err(final_sc_kep_mean[3], final_target_kep_mean[3])
    dW_m = ang_err(final_sc_kep_mean[4], final_target_kep_mean[4])
    dnu_m = ang_err(final_sc_kep_mean[5], final_target_kep_mean[5])
    du_deg_m = ang_err(u_chaser_mean, u_target_mean)
    du_dist_m = abs((u_target_mean - u_chaser_mean + np.pi) % (2 * np.pi) - np.pi) * (
        final_sc_kep_mean[0] / 1e3
    )

    # Calculate Osculating Errors
    da_o = abs(final_sc_kep_osc[0] - final_target_kep_osc[0]) / 1e3
    de_o = abs(final_sc_kep_osc[1] - final_target_kep_osc[1])
    di_o = np.rad2deg(abs(final_sc_kep_osc[2] - final_target_kep_osc[2]))
    dw_o = ang_err(final_sc_kep_osc[3], final_target_kep_osc[3])
    dW_o = ang_err(final_sc_kep_osc[4], final_target_kep_osc[4])
    dnu_o = ang_err(final_sc_kep_osc[5], final_target_kep_osc[5])
    du_deg_o = ang_err(u_chaser_osc, u_target_osc)
    du_dist_o = abs((u_target_osc - u_chaser_osc + np.pi) % (2 * np.pi) - np.pi) * (
        final_sc_kep_osc[0] / 1e3
    )

    # Log the comparative table
    logging.info("--- ABSOLUTE ERRORS ---")
    logging.info(f"{'Element':<12} | {'Mean Error':<18} | {'Osculating Error':<18}")
    logging.info("-" * 55)
    logging.info(f"{'Delta a':<12} | {da_m:.3f} km{'':<9} | {da_o:.3f} km")
    logging.info(f"{'Delta e':<12} | {de_m:.6f}{'':<10} | {de_o:.6f}")
    logging.info(f"{'Delta i':<12} | {di_m:.4f}°{'':<10} | {di_o:.4f}°")
    logging.info(f"{'Delta w':<12} | {dw_m:.4f}°{'':<10} | {dw_o:.4f}°")
    logging.info(f"{'Delta RAAN':<12} | {dW_m:.4f}°{'':<10} | {dW_o:.4f}°")
    logging.info(f"{'Delta nu':<12} | {dnu_m:.4f}°{'':<10} | {dnu_o:.4f}°")
    logging.info(f"{'Delta u':<12} | {du_deg_m:.4f}°{'':<10} | {du_deg_o:.4f}°")
    logging.info(
        f"{'Delta u dist':<12} | {du_dist_m:.3f} km{'':<9} | {du_dist_o:.3f} km\n"
    )


def run_simulation(sim_config: SimulationConfig) -> tuple:
    """
    Executes the Tudat propagation for the given simulation configuration.
    Args:
        - sim_config (SimulationConfig): Configuration object detailing the mission scenario.
    Returns:
        - tuple: A tuple containing (state_array, dependent_variable_array, results_summary_dict).
    """
    # --- Environment ---
    bodies = create_environment(sim_config)

    # --- Instantiate the Guidance and Variables ---
    if sim_config.spacecraft.mode == PropulsionType.ION:
        guidance = IonQLaw(bodies, sim_config)
    else:
        guidance = SailQLaw(bodies, sim_config)

    # --- Models ---
    acceleration_models = create_acceleration_models(bodies, sim_config, guidance)
    mass_rate_models = create_mass_rate_models(
        bodies, sim_config, guidance, acceleration_models
    )

    # --- Termination Settings ---
    # Convergence
    error_variable = propagation_setup.dependent_variable.custom_dependent_variable(
        lambda: get_normalized_target_error(guidance), 1
    )
    target_termination = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=error_variable,
        limit_value=1.0,
        use_as_lower_limit=True,
        terminate_exactly_on_final_condition=False,
    )
    # Max days
    sim_end_epoch = guidance.sim_start_epoch + (
        guidance.param.config.sim_max_days * 86400.0
    )
    time_termination = propagation_setup.propagator.time_termination(sim_end_epoch)
    # Mass dpeletion
    mass_variable = propagation_setup.dependent_variable.body_mass("Spacecraft")
    mass_termination = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=mass_variable,
        limit_value=1.0,
        use_as_lower_limit=True,
        terminate_exactly_on_final_condition=False,
    )
    # Global
    termination_settings = propagation_setup.propagator.hybrid_termination(
        [time_termination, target_termination, mass_termination],
        fulfill_single_condition=True,
    )

    # --- Dependent Variables to Save ---
    dependent_variables_to_save = get_dependent_variable_save_settings(
        sim_config.spacecraft, guidance
    )

    # --- Initial Conditions ---
    initial_cart_sc = element_conversion.keplerian_to_cartesian(
        sim_config.initial_state.values, guidance.param.mu_earth
    )
    initial_cart_target = element_conversion.keplerian_to_cartesian(
        sim_config.target_state.values, guidance.param.mu_earth
    )
    combined_initial_states = np.concatenate((initial_cart_sc, initial_cart_target))

    # --- Integrator Logic ---
    integrator_settings = get_integrator_settings(sim_config, guidance)

    # --- Propagator Settings ---
    propagator_settings = propagation_setup.propagator.multitype(
        propagator_settings_list=[
            propagation_setup.propagator.translational(
                central_bodies=["Earth", "Earth"],
                acceleration_models=acceleration_models,
                bodies_to_integrate=["Spacecraft", "Target"],
                initial_states=combined_initial_states,
                termination_settings=termination_settings,
            ),
            propagation_setup.propagator.mass(
                bodies_with_mass_to_propagate=["Spacecraft"],
                mass_rate_models=mass_rate_models,
                initial_body_masses=np.array([sim_config.spacecraft.initial_mass]),
                termination_settings=termination_settings,
            ),
        ],
        integrator_settings=integrator_settings,
        initial_time=guidance.sim_start_epoch,
        termination_settings=termination_settings,
        output_variables=dependent_variables_to_save,
    )

    # --- Execute Simulation ---
    logging.warning(f"\nStarting Simulation: {sim_config.name}")
    dynamics_simulator = dynamics.simulator.create_dynamics_simulator(
        bodies, propagator_settings
    )

    # --- Return the raw results and copmile them ---
    state_array = result2array(dynamics_simulator.state_history)
    dep_array = result2array(dynamics_simulator.dependent_variable_history)
    final_time = state_array[-1, 0]
    sim_duration_days = (final_time - guidance.sim_start_epoch) / 86400.0

    final_mass = state_array[-1, 13]
    is_success = (
        sim_duration_days < (sim_config.sim_max_days - 0.01) and final_mass > 1.001
    )
    final_sc_kep = shared_math.fast_cart2kep(
        state_array[-1, 1:7], guidance.param.mu_earth
    )
    final_target_kep = guidance.var.target_osc_kep
    final_target_cart = element_conversion.keplerian_to_cartesian(
        final_target_kep, guidance.param.mu_earth
    )

    u_chaser = final_sc_kep[3] + final_sc_kep[5]
    u_target = final_target_kep[3] + final_target_kep[5]
    du_rad = abs((u_chaser - u_target + np.pi) % (2 * np.pi) - np.pi)
    du_deg = np.rad2deg(du_rad)

    final_dist_km = (
        np.linalg.norm(state_array[-1, 1:4] - final_target_cart[0:3]) / 1000.0
    )
    final_vel_ms = np.linalg.norm(state_array[-1, 4:7] - final_target_cart[3:6])

    switch_epoch = getattr(guidance, "stage_2_start_time", np.inf)
    if switch_epoch < np.inf:
        stage_switch_sec = switch_epoch - guidance.sim_start_epoch
    else:
        stage_switch_sec = final_time - guidance.sim_start_epoch

    res_summary = {
        "success": is_success,
        "duration": sim_duration_days,
        "final_du": du_deg,
        "final_distance_km": final_dist_km,
        "final_velocity_ms": final_vel_ms,
        "stage_switch_time": stage_switch_sec,
    }

    # --- Logging in Console ---
    print_termination_outputs(sim_duration_days, sim_config, state_array, guidance)

    return state_array, dep_array, res_summary
