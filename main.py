from config.database import get_mission
from propagator import run_simulation
import plotting

from data_logger import save_simulation_data

import time
import logging

from tudatpy.interface import spice


logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
spice.load_standard_kernels()


def main_simple_propagation(name_scenario: str) -> None:
    """
    Main execution function for the Q-law guidance simulation.
    Args:
        - name_scenario (str): The exact name key of the mission scenario to run.
    Returns:
        - None
    """
    start_time = time.perf_counter()

    sim_config = get_mission(name_scenario)
    state_array, dep_array, res_summary = run_simulation(sim_config)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    logging.warning(
        f"The mission took {elapsed_time:.2f} seconds to run. This a value of {elapsed_time/res_summary['duration']:.3f} seconds per simulated day."
    )

    try:
        save_simulation_data(sim_config, state_array, dep_array, res_summary)
    except Exception as e:
        logging.error(f"Failed to save simulation data: {e}")

    plotting.plot_mission(state_array, dep_array, sim_config)


if __name__ == "__main__":

    main_simple_propagation("Phase_Sweep")
