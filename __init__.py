import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import logging


# =============================================================================
# 0. CONSTANTS
# =============================================================================

DEFAULT_MU_EARTH = 3.986004418e14
DEFAULT_P_1AU = 1366.1 / 299792458.0
DEFAULT_AU = 149597870700.0
DEFAULT_G0 = 9.80665
DEFAULT_EARTH_RADIUS = 6378137.0


# =============================================================================
# 1. ENUMS (Definitions)
# =============================================================================


class PropulsionType(Enum):
    """Enumeration of supported spacecraft propulsion systems."""

    ION = auto()
    SAIL = auto()


class CoordinateType(Enum):
    """Enumeration of supported orbital coordinate types."""

    CARTESIAN = auto()
    KEPLERIAN = auto()


class SailForceModel(Enum):
    """Enumeration of supported solar sail optical force models."""

    IDEAL = auto()


class IntegratorType(Enum):
    """Enumeration of supported numerical integration schemes."""

    RK4 = auto()
    RKF78 = auto()


class ReferenceFrame(Enum):
    """Enumeration of supported spatial reference frames."""

    J2000 = auto()
    ECLIPJ2000 = auto()


class GravityModel(Enum):
    """Enumeration of supported Earth gravity field models."""

    POINT_MASS = auto()
    J2 = auto()


class EclipseModel(Enum):
    """Enumeration of supported eclipse/shadow models."""

    NONE = auto()
    CONICAL = auto()


# =============================================================================
# 2. COMPONENT CONFIGS
# =============================================================================


@dataclass
class StateDefinition:
    """
    Associates a state vector with its coordinate type and reference frame.

    Attributes:
        - values (np.ndarray): The 6-element state vector array.
        - type (CoordinateType): The coordinate system of the values (e.g., Cartesian, Keplerian).
        - frame (ReferenceFrame): The reference frame of the values (e.g., J2000).
    """

    values: np.ndarray
    type: CoordinateType
    frame: ReferenceFrame = ReferenceFrame.J2000  # Defaults to J2000 if not specified

    def __post_init__(self):
        if isinstance(self.values, list):
            self.values = np.array(self.values)
        if self.values.shape != (6,):
            raise ValueError(
                f"State vector must have 6 elements. Got {self.values.shape} elements."
            )


@dataclass
class IntegratorConfig:
    """
    Settings for the Tudat numerical integrator.

    Attributes:
        - type (IntegratorType): The integration algorithm to use.
        - initial_step_size (float): Starting step size in seconds.
        - min_step_size (float): Minimum allowable step size for variable integrators.
        - max_step_size (float): Maximum allowable step size for variable integrators.
        - rel_tol (float): Relative error tolerance.
        - abs_tol (float): Absolute error tolerance.
    """

    type: IntegratorType = IntegratorType.RKF78
    initial_step_size: float = 60.0

    # Parameters for variable step-size integrators (like RKF78)
    min_step_size: float = 1e-3
    max_step_size: float = 1200.0
    rel_tol: float = 1.0e-12
    abs_tol: float = 1.0e-12


@dataclass
class SpacecraftConfig:
    """
    Physical parameters and propulsion settings of the spacecraft.

    Attributes:
        - mode (PropulsionType): The type of propulsion (Ion or Sail).
        - initial_mass (float): Starting mass of the spacecraft in kg.
        - model (SailForceModel | None): The optical force model for a solar sail.
        - area_to_mass (float): The sail's area-to-mass ratio in m^2/kg.
        - mass_depletion (bool): Flag to enable/disable fuel consumption for ion engines.
        - thrust (float): Engine thrust magnitude in Newtons.
        - isp (float): Engine specific impulse in seconds.
    """

    mode: PropulsionType
    initial_mass: float  # kg

    # Sail specific
    model: Optional[SailForceModel] = None
    area_to_mass: float = 0.0  # m2/kg
    mass_depletion: bool = True  # if false, mass will not decrease

    # Ion specific
    thrust: float = 0.0  # N
    isp: float = 0.0  # s

    def __post_init__(self):
        if self.initial_mass <= 0:
            raise ValueError("Spacecraft mass must be positive.")

        if self.mode == PropulsionType.ION:
            if self.thrust <= 0 or self.isp <= 0:
                raise ValueError("For ION mode, 'thrust' and 'isp' must be positive.")
            if self.area_to_mass > 0 or self.model is not None:
                logging.warning(
                    "Sail parameters (area/model) defined but mode is ION. They will be ignored."
                )

        elif self.mode == PropulsionType.SAIL:
            if self.area_to_mass <= 0:
                logging.warning("The Area to mass ratio was set to 0.")
            if self.model is None:
                self.model = SailForceModel.IDEAL
                logging.info("No Sail Force Model selected. Defaulting to IDEAL.")

            if self.thrust > 0 or self.isp > 0:
                logging.warning(
                    "Thrust or ISP defined but mode is SAIL. Engine parameters will be ignored."
                )


@dataclass
class QLawConfig:
    """
    Tuning parameters and penalty weights for the Q-Law guidance algorithm.

    Attributes:
        - weights (np.ndarray): Penalty weights for targeting [a, e, i, w, W, nu].
        - eta_cut (float): Efficiency threshold below which thrust is cut off.
        - m (float): Scaling parameter for semi-major axis error.
        - n (float): Power scaling parameter for semi-major axis error.
        - r (float): Root scaling parameter for semi-major axis error.
        - rp_min (float): Minimum allowable periapsis radius in meters (collision avoidance).
        - k_imp (float): Penalty growth rate for approaching rp_min.
        - w_imp (float): Activation weight for the periapsis penalty.
        - a_max_limit (float): Maximum allowable semi-major axis in meters (escape avoidance).
        - k_esc (float): Penalty growth rate for approaching a_max_limit.
        - w_esc (float): Activation weight for the escape penalty.
        - W_L (float): Phasing penalty weight for rendezvous targeting.
        - W_scl (float): Phasing scaling factor for rendezvous targeting.
    """

    weights: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    )

    eta_cut: float = 0.0

    # Scaling parameters
    m: float = 3.0
    n: float = 4.0
    r: float = 2.0

    # Penalty Parameters
    rp_min: float = 6578000.0
    k_imp: float = 100.0
    w_imp: float = 0.0
    a_max_limit: float = 1.0e9
    k_esc: float = 0.0
    w_esc: float = 0.0

    # For rendezvous
    W_L: float = 0.1
    W_scl: float = 0.7


@dataclass
class LoggingConfig:
    """
    Configuration for terminal logging outputs, data saving, and plotting.

    Attributes:
        - plot_elements (bool): Flag to generate orbital element plots.
        - plot_trajectory_3d (bool): Flag to generate a 3D trajectory plot.
        - plot_rendezvous_zoom_3d (bool): Flag to generate a zoomed 3D rendezvous plot.
        - plot_thrust (bool): Flag to generate a thrust profile plot.
        - plot_mass (bool): Flag to generate a mass depletion plot.
        - plot_control_angles (bool): Flag to generate control angle plots.
        - plot_aspect_angle (bool): Flag to generate an aspect angle plot.
        - plot_q_history (bool): Flag to generate a total Q-value history plot.
        - plot_q_terms (bool): Flag to generate individual Q-term breakdown plots.
        - plot_rendezvous_convergence (bool): Flag to generate a rendezvous convergence plot.
        - plot_eclipse (bool): Flag to generate an eclipse/shadow plot.
    """

    log_step_size: float = 86400.0  # Log every 24 hours by default
    log_level: int = logging.INFO

    # What to plot? (Data is always saved, these just toggle the visual outputs)
    plot_elements: bool = False
    plot_trajectory_3d: bool = False
    plot_rendezvous_zoom_3d: bool = False
    plot_thrust: bool = False
    plot_mass: bool = False
    plot_control_angles: bool = False
    plot_aspect_angle: bool = False
    plot_q_history: bool = False
    plot_q_terms: bool = False
    plot_rendezvous_convergence: bool = False
    plot_thrust_components: bool = False
    plot_eclipse: bool = False
    plot_errors: bool = False


# =============================================================================
# 3. THE CENTRALIZED CONFIG OBJECT
# =============================================================================


@dataclass
class SimulationConfig:
    """
    The centralized master configuration object defining a complete mission scenario.

    Attributes:
        - name (str): Identifier name for the mission scenario.
        - spacecraft (SpacecraftConfig): Spacecraft physical and propulsion setup.
        - initial_state (StateDefinition): The starting state of the chaser spacecraft.
        - target_state (StateDefinition): The desired end state or target spacecraft state.
        - guidance (QLawConfig): Settings for the Q-Law control algorithm.
        - logging (LoggingConfig): Settings for data recording and visualization.
        - integrator (IntegratorConfig): Settings for numerical propagation.
        - is_rendezvous (bool): Flag indicating if 6-element targeting (phasing) is active.
        - earth_gravity_model (GravityModel): The fidelity of the Earth gravity field.
        - eclipse_model (EclipseModel): The fidelity of the eclipse/shadow model.
        - stages (List[int]): The sequence of active stages for rendezvous (e.g., [1, 2, 3]).
        - stage_1_tol (float): Tolerance for stage 1 (far-range) in terms of relative error.
        - stage_1_tol_angle (float): Tolerance for stage 1 in terms of angular error in radians.
        - stage_2_tol (float): Tolerance for stage 2 (mid-range) in terms of relative error.
        - stage_2_tol_angle (float): Tolerance for stage 2 in terms of angular error in radians.
        - stage_3_tol_distance (float): Tolerance for stage 3 (close-range) in terms of distance in kilometers.
        - stage_3_tol_velocity (float): Tolerance for stage 3 in terms of velocity in m/s.
        - sim_start_epoch (float): Mission start time in seconds since J2000.
        - sim_max_days (float): Hard time limit for the simulation in days.
    """

    name: str
    spacecraft: SpacecraftConfig
    initial_state: StateDefinition
    target_state: StateDefinition
    guidance: QLawConfig
    logging: LoggingConfig

    integrator: IntegratorConfig = field(default_factory=IntegratorConfig)

    is_rendezvous: bool = False  # DEfault settings

    earth_gravity_model: GravityModel = GravityModel.POINT_MASS
    eclipse_model: EclipseModel = EclipseModel.NONE

    # Staging
    stages: list[int] = field(default_factory=lambda: [1, 2, 3])

    stage_1_tol: float = 2e-3  # 0.2% default
    stage_1_tol_angle: float = np.deg2rad(0.5)  # 0.5° default
    stage_2_tol: float = 1e-4  # 0.2% is 14 km error in a_0=7078km orbit
    stage_2_tol_angle: float = np.deg2rad(0.3)  # 0.3° is 40 km error if a_0=7078km
    stage_3_tol_distance: float = 10.0  # 10km default
    stage_3_tol_velocity: float = 10  # 0.01 m/s defualt

    sim_start_epoch: float = 0.0
    sim_max_days: float = 100.0  # 100 days simulation by default

    def __post_init__(self):
        from guidance.math_utils.shared_math import standardize_state_vector

        target_a = self.target_state.values[0]
        target_e = self.target_state.values[1]
        target_rp = target_a * (1.0 - target_e)
        initial_a = self.initial_state.values[0]
        initial_e = self.initial_state.values[1]

        standardize_state_vector(
            self.initial_state, DEFAULT_MU_EARTH, self.sim_start_epoch
        )
        standardize_state_vector(
            self.target_state, DEFAULT_MU_EARTH, self.sim_start_epoch
        )

        num_weights = len(self.guidance.weights)

        # ============================================================================
        # Error Handling
        if num_weights != 6:
            raise ValueError(
                f"Configuration Error: The engine requires 6 weights, one for each orbital element: [a, e, i, w, W, nu].\n"
                f"-> Got {num_weights} weights."
            )
        if target_rp < self.guidance.rp_min:
            raise ValueError(
                "\n[ERROR] Target orbit is invalid! Target orbit periapsis is below the rp_min safety threshold.\n"
            )
        if initial_a * (1.0 - initial_e) < self.guidance.rp_min:
            raise ValueError(
                "\n[ERROR] Initial orbit is invalid! Initial orbit periapsis is below the rp_min safety threshold."
            )
        # ============================================================================

        # ============================================================================
        # Initialization Terminal Output
        init_vals_str = (
            "[" + ", ".join([f"{v:.2f}" for v in self.initial_state.values]) + "]"
        )
        orig_init_str = f"{self.initial_state.type.name} in {self.initial_state.frame.name} = {init_vals_str}"
        tgt_vals_str = (
            "[" + ", ".join([f"{v:.2f}" for v in self.target_state.values]) + "]"
        )
        orig_tgt_str = f"{self.target_state.type.name} in {self.target_state.frame.name} = {tgt_vals_str}"

        logging.info(f"--- Configuration '{self.name}' Loaded Successfully ---")
        logging.info(f"    Mode: {self.spacecraft.mode.name}")
        logging.info(f"    Rendezvous: {self.is_rendezvous}")
        logging.info(f"    Initial State: {orig_init_str}")
        logging.info(f"    Target State: {orig_tgt_str}")
        # ============================================================================


@dataclass
class SimulationParameters:
    """
    Fixed parameters and physical constants used throughout the simulation.

    Attributes:
        - config (SimulationConfig): The immutable mission configuration object.
        - mu_earth (float): Earth's gravitational parameter in m^3/s^2.
        - earth_radius (float): Earth's equatorial radius in meters.
        - g0 (float): Standard gravity in m/s^2.
        - AU (float): Astronomical Unit in meters.
        - P_1AU (float): Solar radiation pressure at 1 AU in N/m^2.
        - mu_sun (float): Sun's gravitational parameter in m^3/s^2.
    """

    config: SimulationConfig

    mu_earth: float = DEFAULT_MU_EARTH  # m3/s2
    earth_radius: float = DEFAULT_EARTH_RADIUS  # m
    g0: float = DEFAULT_G0  # m/s2
    AU: float = DEFAULT_AU  # m
    P_1AU: float = DEFAULT_P_1AU  # N/m2

    mu_sun: float = 1.32712440042e20  # m3/s2



@dataclass
class SimulationVariables:
    """
    Dynamically varying quantities updated at every integration step.

    Attributes:
        - time (float): Current simulation epoch in seconds.
        - cartesian_state (np.ndarray): Current spacecraft state in Cartesian coordinates.
        - keplerian_state (np.ndarray): Current spacecraft state in Keplerian elements.
        - mass (float): Current spacecraft mass in kg.
        - s_vec_eci (np.ndarray): Unit vector pointing from spacecraft to the Sun in ECI.
        - max_rates (np.ndarray): Current maximum rates of change for the orbital elements.
        - thrust_vector (np.ndarray): Current commanded thrust vector.
        - current_stage (int): Current active phase of the rendezvous (1 or 2).
        - stage_switch_time (float): Epoch when the simulation transitioned to stage 2.
        - updated_target_keplerian_state (np.ndarray): The target's current propagated Keplerian elements.
    """

    time: float = 0.0
    mass: float = 0.0

    chaser_osc_kep: np.ndarray = field(default_factory=lambda: np.zeros(6))
    chaser_mean_kep: np.ndarray = field(default_factory=lambda: np.zeros(6))
    target_osc_kep: np.ndarray = field(default_factory=lambda: np.zeros(6))
    target_mean_kep: np.ndarray = field(default_factory=lambda: np.zeros(6))

    s_vec_eci: np.ndarray = field(default_factory=lambda: np.zeros(6))

    max_rates: np.ndarray = field(default_factory=lambda: np.zeros(5))
    thrust_vector: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # For rendezvous
    current_stage: int = 0
