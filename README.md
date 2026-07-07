# Solar Sail Rendezvous Q-Law Framework

This framework is a Python-based astrodynamics simulation environment designed to optimize and evaluate continuous-thrust orbital maneuvers for solar sails. It is based on the Q-Law (a Lyapunov feedback control algorithm) to calculate real-time thrust vectors for both Ion Propulsion and Solar Sail spacecraft. 

Under the hood, the framework uses Tudat (tudatpy) as its high-fidelity physics engine to propagate the spacecraft's state over time, while employing Numba to ensure the guidance algorithms run efficiently.

---

## Required Libraries

To run this framework and its accompanying interactive dashboards, you need a Python environment with the following packages installed:
*   **`tudatpy`**: The core astrodynamics and numerical integration engine.
*   **`numpy`**: For matrix and vector operations.
*   **`numba`**: To compile the mathematical guidance functions (`@njit`) to fast machine code.
*   **`pandas`**: To structure the telemetry data before exporting.
*   **`pyarrow`** or **`fastparquet`**: Engine required by pandas to save data as `.parquet` files.
*   **`plotly`**: To generate interactive, web-based 3D trajectories and data plots.
*   **`streamlit`**: To run the interactive multi-run analysis and phase sweep web dashboards.

---

## Codebase Structure

The framework is modular, separating the configuration, mathematics, physics engine, and data processing into distinct layers:

### 1. Configuration (`__init__.py` & `database.py` in the config folder)
*   **`__init__.py`**: Defines the data structures (`dataclasses`) and options (`Enums`) for the simulation. It holds configurations for the spacecraft properties, Q-Law weights, numerical integrator, and logging preferences.
*   **`database.py`**: Acts as a library of specific mission scenarios (e.g., `build_phase_sweep`, `build_test_case`). It instantiates the configurations defined in `config.py` and returns a complete `SimulationConfig` object.

### 2. Guidance & Mathematics (`guidance/`)
*   **`base_q_law.py`**: The abstract base class that tracks the simulation state, computes rolling mean elements to filter out J2 perturbation noise, and manages stage transitions for rendezvous missions.
*   **`q_law_ion.py` & `q_law_sail.py`**: Subclasses that inherit from `BaseQLaw`. They compute the specific acceleration vector required at any given timestamp based on the propulsion type.
*   **`math_utils/` (`shared_math.py`, `ion_math.py`, `sail_math.py`)**: Pure mathematical functions (Gauss Variational Equations, coordinate transformations, control angle calculations) wrapped in `@njit(cache=True, fastmath=True)` for maximum performance.

### 3. Physics Propagation (`propagator.py`)
This module is the bridge to Tudat. It reads your `SimulationConfig` and sets up the Earth/Sun environment, custom acceleration models (linking Tudat to your Q-Law output), mass depletion, and termination conditions (e.g., reaching the target or running out of time).

### 4. Post-Processing & Telemetry (`data_logger.py` & `plotting.py`)
*   **`data_logger.py`**: Extracts the history of the spacecraft and target from Tudat, computes physical metrics (like thrust allocations and true anomalies), and saves everything into a timestamped folder inside `results/` as a `config.json` and a `telemetry.parquet` file.
*   **`plotting.py`**: Reads the telemetry arrays and generates an interactive HTML dashboard containing 3D orbits, orbital element histories, Q-value drops, and convergence plots immediately after a simulation runs.

### 5. Interactive Dashboards (Streamlit Modules)
Two dedicated Streamlit modules are available for post-simulation analysis across multiple data sets:
*   **SWEEP Telemetry Explorer**: A web app that navigates the `results/` directory to load and compare `.parquet` telemetry files from multiple simulation runs simultaneously. It features dynamic unit conversion (e.g., Radians to Degrees, Meters to Kilometers), customizable subplots, and adjustable smoothing windows.
*   **Phase Sweep Explorer**: A web app designed specifically to analyze convergence trends from Phase Sweep missions saved as `.csv` files. It visualizes Time of Flight (ToF) metrics against initial phase offsets, and includes tools for data normalization, polynomial curve fitting, and timeout boundary visualization.

---

## How It Works (The Simulation Loop)

For a user with no Tudat experience, the core simulation loop operates as follows:
1.  **Initialization:** `main.py` requests a specific mission from `database.py`.
2.  **Environment Setup:** `propagator.py` spawns a virtual Earth, a virtual Sun, the "Chaser" spacecraft, and a dummy "Target" object in Tudat.
3.  **The Loop:** Tudat's numerical integrator moves time forward step-by-step. At every step:
    *   Tudat asks the active `Guidance` class (Ion or Sail) for the current acceleration.
    *   The `Guidance` class looks at the current and target orbital elements.
    *   Using the mathematical gradients from `math_utils`, it figures out the optimal direction to thrust to minimize the Lyapunov "Q" penalty function.
    *   It returns an acceleration vector (in the ECI frame) back to Tudat.
    *   Tudat applies this acceleration, updates the spacecraft's speed, position, and mass, and advances to the next timestamp.
4.  **Termination:** The loop stops when the physical error between the Chaser and Target drops below your defined tolerances, or the simulation reaches its maximum duration.

---

## Understanding Enums and Modifying the Code

Enums (Enumerations) are strictly typed lists of options defined in `config.py`. They prevent spelling errors and make it easy to route logic. 

**Current Enums:**
*   `PropulsionType`: `ION`, `SAIL`.
*   `CoordinateType`: `CARTESIAN`, `KEPLERIAN`.
*   `SailForceModel`: `IDEAL`.
*   `IntegratorType`: `RK4`, `RKF78`.
*   `ReferenceFrame`: `J2000`, `ECLIPJ2000`.
*   `GravityModel`: `POINT_MASS`, `J2`.
*   `EclipseModel`: `NONE`, `CONICAL`.

### Example: Adding a New Sail Force Model (e.g., `OPTICAL`)

If you want to add a new physical model for the solar sail (one that accounts for reflection and absorption), follow these steps to cleanly extend the framework:

1.  **Update the Enum (`config.py`):**
    Add the new option to the `SailForceModel` enum.
    ```python
    class SailForceModel(Enum):
        IDEAL = auto()
        OPTICAL = auto() # <-- Add this
    ```

2.  **Add the Math (`sail_math.py`):**
    Create a constant identifier and implement the new mathematical logic inside `get_acceleration_vector_s` and `compute_optimal_angles_sail`.
    ```python
    IDEAL = 0
    OPTICAL = 1 # <-- Add this

    @njit(...)
    def get_acceleration_vector_s(..., model_id: int, ...):
        # Existing logic
        if model_id == IDEAL:
            a_s = ... 
        elif model_id == OPTICAL:
            # <-- Add your custom physics equation here
            a_s = ... 
        return a_s
    ```

3.  **Update the Router (`q_law_sail.py`):**
    Map your new enum to the math identifier in the `__init__` method.
    ```python
    if model_enum == SailForceModel.IDEAL:
        self.model_id = sail_math.IDEAL
    elif model_enum == SailForceModel.OPTICAL:
        self.model_id = sail_math.OPTICAL # <-- Add this routing
    ```

4.  **Use it in a Mission (`database.py`):**
    Change the spacecraft config in your mission builder.
    ```python
    sc_config = SpacecraftConfig(
        mode=PropulsionType.SAIL,
        model=SailForceModel.OPTICAL, # <-- Now you can use it!
        ...
    )
    ```

---

## Quickstart: Running the Framework

**1. Running a Simulation:**
*   Open `database.py` and review/create a mission building function (e.g., `build_phase_sweep()`). 
*   Ensure it is registered in the `MISSION_CATALOG` dictionary at the bottom of the file.
*   Open `main.py` and pass the exact string name of your mission to the execution function:
    ```python
    if __name__ == "__main__":
        main_simple_propagation("Phase_Sweep")
    ```
*   Run `python main.py` in your terminal. The console will print daily summaries, and interactive Plotly charts will open automatically in your browser when finished.

**2. Running the Interactive Dashboards:**
*   To explore your `.parquet` telemetry data across multiple runs, launch the Telemetry Explorer in your terminal:
    ```bash
    streamlit run visualizer_simulation.py
    ```
*   To analyze convergence trends from sweep `.csv` files, launch the Phase Sweep Explorer:
    ```bash
    streamlit run visualizer_sweep.py
    ```
