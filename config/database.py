import numpy as np
from tudatpy.interface import spice
from tudatpy.astro import element_conversion

from config import (
    SimulationConfig,
    SpacecraftConfig,
    StateDefinition,
    QLawConfig,
    LoggingConfig,
    IntegratorConfig,
    PropulsionType,
    CoordinateType,
    IntegratorType,
    SailForceModel,
    DEFAULT_MU_EARTH,
    DEFAULT_EARTH_RADIUS,
    DEFAULT_P_1AU,
    ReferenceFrame,
    GravityModel,
    EclipseModel,
)

# ============================================================================
# 1. MISSION BUILDERS (One clean function per mission)
# ============================================================================


def build_phase_sweep() -> SimulationConfig:
    sma_0 = DEFAULT_EARTH_RADIUS + 700.0e3
    initial_state = StateDefinition(
        values=np.array(
            [
                sma_0 + 100e3,
                0.0,
                np.deg2rad(98.19),
                0.0,
                np.deg2rad(289.38),
                0.0,
            ]
        ),
        type=CoordinateType.KEPLERIAN,
        frame=ReferenceFrame.J2000,
    )

    target_state = StateDefinition(
        values=np.array(
            [
                sma_0,
                0.0,
                np.deg2rad(98.19),
                0.0,
                np.deg2rad(289.38),
                0.0,
            ]
        ),
        type=CoordinateType.KEPLERIAN,
        frame=ReferenceFrame.J2000,
    )

    sc_config = SpacecraftConfig(
        mode=PropulsionType.SAIL,
        initial_mass=15.0,
        thrust=1.5e-3,
        isp=3000,
        mass_depletion=False,
        model=SailForceModel.IDEAL,
        area_to_mass=1e-4 / (2 * DEFAULT_P_1AU),
    )

    integrator_config = IntegratorConfig(
        type=IntegratorType.RK4,
        initial_step_size=60,
        min_step_size=1e-3,
        max_step_size=1200.0,
        rel_tol=1e-8,
        abs_tol=1e-8,
    )

    qlaw_config = QLawConfig(
        weights=np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        eta_cut=0.0,
        m=3.0,
        n=4.0,
        r=2.0,
        w_imp=1.0,
        k_imp=100.0,
        rp_min=DEFAULT_EARTH_RADIUS + 200e3,
        w_esc=1.0,
        k_esc=100.0,
        a_max_limit=1.0e9,
        W_L=0.1,
        W_scl=1.0,
    )

    logging_config = LoggingConfig(
        log_step_size=86400.0,
        plot_trajectory_3d=True,
        plot_elements=True,
        plot_rendezvous_convergence=True,
        plot_q_terms=True,
        plot_thrust_components=False,
        plot_aspect_angle=True,
        plot_errors=True,
    )

    return SimulationConfig(
        name="Phase_Sweep",
        spacecraft=sc_config,
        initial_state=initial_state,
        target_state=target_state,
        earth_gravity_model=GravityModel.J2,
        eclipse_model=EclipseModel.CONICAL,
        guidance=qlaw_config,
        logging=logging_config,
        integrator=integrator_config,
        is_rendezvous=True,
        stages=[1],
        stage_1_tol=4e-3,  # SMA and e
        stage_1_tol_angle=np.deg2rad(0.5),
        stage_2_tol=4e-3,  # SMA and e
        stage_2_tol_angle=np.deg2rad(0.5),  # u
        stage_3_tol_distance=12.0,
        stage_3_tol_velocity=12.0,
        sim_start_epoch=spice.convert_date_string_to_ephemeris_time(
            "2022-10-14 16:09:34 UTC"
        ),
        sim_max_days=5.0,
    )


def build_test_case() -> SimulationConfig:
    sma_target = DEFAULT_EARTH_RADIUS + 700.0e3
    initial_state = StateDefinition(
        values=np.array(
            [
                sma_target - 100e3,
                0.0,
                np.deg2rad(0.5),
                0.0,
                np.deg2rad(289.392),
                0,
            ]
        ),
        type=CoordinateType.KEPLERIAN,
        frame=ReferenceFrame.J2000,
    )

    target_state = StateDefinition(
        values=np.array(
            [
                sma_target,
                0.0,
                np.deg2rad(0.5),
                0.0,
                np.deg2rad(289.392),
                np.deg2rad(-34),
            ]
        ),
        type=CoordinateType.KEPLERIAN,
        frame=ReferenceFrame.J2000,
    )

    a_c = 0  # 1e-4
    sc_config = SpacecraftConfig(
        mode=PropulsionType.SAIL,
        initial_mass=15.0,
        thrust=0.0,
        isp=0.0,
        mass_depletion=False,
        model=SailForceModel.IDEAL,
        area_to_mass=a_c / (2 * DEFAULT_P_1AU),
    )

    integrator_config = IntegratorConfig(
        type=IntegratorType.RK4,
        initial_step_size=60,
        min_step_size=1e-3,
        max_step_size=1200.0,
        rel_tol=1e-8,
        abs_tol=1e-8,
    )

    qlaw_config = QLawConfig(
        weights=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        eta_cut=0.0,
        m=3.0,
        n=4.0,
        r=2.0,
        w_imp=1.0,
        k_imp=100.0,
        rp_min=DEFAULT_EARTH_RADIUS + 200e3,
        w_esc=0.0,
        k_esc=0.0,
        a_max_limit=1.0e9,
        W_L=0.17,
        W_scl=1.4,
    )

    logging_config = LoggingConfig(
        log_step_size=86400.0,
        plot_trajectory_3d=False,
        plot_control_angles=True,
        plot_thrust=False,
        plot_elements=True,
        plot_rendezvous_zoom_3d=False,
        plot_rendezvous_convergence=True,
        plot_q_terms=True,
        plot_aspect_angle=True,
        plot_phase_plane_portrait=False,
        plot_thrust_components=True,
        plot_q_history=False,
        plot_efficiency=False,
        plot_eclipse=True,
    )

    return SimulationConfig(
        name="Test_Case",
        spacecraft=sc_config,
        initial_state=initial_state,
        target_state=target_state,
        earth_gravity_model=GravityModel.J2,
        eclipse_model=EclipseModel.NONE,
        guidance=qlaw_config,
        logging=logging_config,
        integrator=integrator_config,
        is_rendezvous=True,
        stages=[1],
        stage_1_tol=4e-3,
        stage_1_tol_angle=np.deg2rad(0.5),
        stage_2_tol=1e-4,
        stage_2_tol_angle=np.deg2rad(0.3),
        stage_3_tol_distance=10.0,
        stage_3_tol_velocity=10.0,
        sim_start_epoch=spice.convert_date_string_to_ephemeris_time(
            "2022-10-14 16:19:01 UTC"
        ),
        sim_max_days=10.0,
    )


# ============================================================================
# 2. THE ROUTING TABLE
# ============================================================================

MISSION_CATALOG = {
    "Phase_Sweep": build_phase_sweep,
    "Test_Case": build_test_case,
}


# ============================================================================
# 3. THE LOADER
# ============================================================================


def get_mission(name: str) -> SimulationConfig:
    """
    Retrieves and instantiates a mission configuration from the catalog.
    Args:
        - name (str): The exact name key of the mission to load.
    Returns:
        - SimulationConfig: The generated mission configuration object.
    """
    if name not in MISSION_CATALOG:
        available = "\n - ".join(MISSION_CATALOG.keys())
        raise ValueError(
            f"\n\n[ERROR] Mission '{name}' not found in database.\n"
            f"Available missions are:\n - {available}\n"
            f"Please check your spelling in main.py!\n"
        )
    return MISSION_CATALOG[name]()
