import json
import logging
import os
import numpy as np
from os.path import join
import sys

# Import custom modules from the same package
path = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(path)
from src.ml_model import get_prediction, plot_prediction
from src.calculate_pf import get_power_data, plot_power_data, run_power_flow
from src.common import RESULTS_DIR
from src.transform_split_data import get_data_for_training
from src.config import N_FEATURES, N_SPLITS, STATUS
from src.results_eval_heatmap import plot_heatmap

logger = logging.getLogger()
# set logging level as INFO 
logger.setLevel(logging.INFO)



def run_model(config):
    """Runs the machine learning model using the provided config.

    Args:
        config (dict): Configuration parameters for the model.

    Returns:
        dict: Results of the model predictions and plots.
    """
    logging.info(f"Evaluating model with params: {config}")
    X_train, X_test, y_train, y_test = get_data_for_training(config)
    y_test, y_predict = get_prediction(X_train, X_test, y_train, y_test)
    results = plot_prediction(y_test, y_predict, config)
    return results

def sweep_ml_model():
    """Run the sweep analysis to evaluate different configurations of the model.

    Returns:
        list: List of results for each configuration.
    """
    plot_power_data()
    logging.info(f"Running the analysis for the {STATUS} configuration")
    config_list = []

    for n in range(6, N_FEATURES):
        for sp in np.linspace(0.5, 0.95, N_SPLITS):
            for pd in [True, False]:
                config = {"n_params": n, "split_val": sp, "param_date": pd}
                config_list.append(config)
    res_list=[]        
    for config in config_list:
        res = run_model(config)
        res_list.append(res)

    # Save the results to the results.json file
    result_filename = join(RESULTS_DIR, f"results_{STATUS[0:3]}.json")
    with open(result_filename, 'w') as f:
        json.dump(res_list, f)
    logging.info(f"Results saved to \n{result_filename}")

    plot_heatmap(result_filename=result_filename)
    return res_list
    

