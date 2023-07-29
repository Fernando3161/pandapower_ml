import os
import random
import sys
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from os.path import join
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Adding path for common functions
path = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(path)

# Importing from local packages
from src.common import FIG_RESULTS_DIR, check_and_create_folders
from src.transform_split_data import get_data_for_training

sns.set_style("darkgrid")
sns.set(font_scale=1)


def get_prediction(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray,
                   y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get predictions using a Multi-Layer Perceptron Regressor.

    Args:
        X_train (np.ndarray): Training input features.
        X_test (np.ndarray): Test input features.
        y_train (np.ndarray): Training target labels.
        y_test (np.ndarray): Test target labels.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing the true test labels and the predicted labels.
    """
    random.seed(123456)
    scaler = StandardScaler()
    y_train = scaler.fit_transform(y_train)
    ann = MLPRegressor(verbose=0)
    ann.fit(X_train, y_train)
    y_predict = ann.predict(X_test)
    y_predict = scaler.inverse_transform(y_predict)

    return y_test, y_predict


def plot_prediction(y_test: np.ndarray, y_predict: np.ndarray,
                    config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Plot the comparison of test and prediction values for multiple lines.

    Args:
        y_test (np.ndarray): True test labels.
        y_predict (np.ndarray): Predicted labels.
        config (Dict[str, Any], optional): Configuration parameters. Defaults to None.

    Returns:
        Dict[str, Any]: Dictionary containing the results and configuration parameters.
    """
    n_params = 6
    split_val = 0.8
    param_date = True

    if config:
        n_params = config["n_params"]
        split_val = config["split_val"]
        param_date = config["param_date"]

    date_features = ""
    if param_date:
        date_features = "_dt_feats"

    tail = f"{n_params}p_{split_val * 100:.0f}s" + date_features
    FOLDER = join(FIG_RESULTS_DIR, tail)
    check_and_create_folders([FOLDER])

    fig, axs = plt.subplots(2, 2, figsize=(14, 6))
    ax_ = [axs[0][0], axs[0][1], axs[1][0], axs[1][1]]

    for i in range(0, 4):
        ax = ax_[i]
        test = [y for y in y_test[i]]
        predict = [y[i] for y in y_predict]
        if len(test) > 24 * 14:
            test = test[:24 * 14]
            predict = predict[:24 * 14]
        data = pd.DataFrame.from_dict({
            "Test Data": test,
            "Prediction": predict})
        data.plot(ax=ax)
        ax.set_xlabel("Point", fontsize=8)
        ax.set_ylabel("Line Load [%]", fontsize=8)
        ax.set_title(f"Line {i}", fontweight="bold", fontsize=9)
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(8)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(8)
    fig.suptitle("Comparison of Test and Prediction values",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(join(FOLDER, f"Test-v-Prediction.png"), dpi=600)
    plt.close("all")

    fig, axs = plt.subplots(2, 2, figsize=(14, 7))
    ax_ = [axs[0][0], axs[0][1], axs[1][0], axs[1][1]]

    sns.set(font_scale=0.7)
    for i in range(0, 4):
        test = [y for y in y_test[i]]
        predict = [y[i] for y in y_predict]
        if len(test) > 24 * 14:
            test = test[:24 * 14]
            predict = predict[:24 * 14]
        ax = ax_[i]

        data_line = pd.DataFrame.from_dict({"x": np.linspace(0, max(test), 100),
                                            "Optimal": np.linspace(0, max(test), 100)})
        data_line.plot.line(x="x", y="Optimal", color="k", ax=ax, style="-")

        import statsmodels.api as sm
        X = test
        y = predict
        X = sm.add_constant(X)  # Add a constant term for the intercept

        # Create and fit the linear regression model
        model = sm.OLS(y, X)
        results = model.fit()

        # Get the parameters
        intercept = results.params[0]
        slope = results.params[1]
        data_trend = pd.DataFrame.from_dict({"x": np.linspace(0, max(test), 100),
                                             "Trend": [intercept + slope * x for x in np.linspace(0, max(test), 100)]})

        data_trend.plot.line(x="x", y="Trend", color="b", ax=ax, style="-.")

        dev = [(test[i] - predict[i])/test[i] for i in range(len(predict))]
        dev = [(abs(d))**2 for d in dev]
        total_dev = sum(dev)**0.5

        data = {"Test": test,
                "Prediction": predict,
                "SDev": dev}

        data_df = pd.DataFrame.from_dict(data)

        sns.scatterplot(data=data_df, x="Test", y="Prediction",
                        hue="SDev", ax=ax, palette="viridis")
        ax.set_xlabel("Test Values [%]", fontsize=8)

        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(8)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(8)

        ax.set_ylabel("Prediction Values [%]", fontsize=8)
        ax.set_title(f"Loading of Line {i}", fontsize=9, fontweight="bold")

        text = f"Slope = {slope:.2f}" + "\n" + f"Error = {total_dev:.2f}"

        ax.legend(loc="upper left")
        ax.text(0.8 * max(test), 0.1 * max(test), text, ha='left')

    fig.suptitle("Comparison of Test and Prediction Values", fontweight="bold")
    fig.tight_layout()
    fig.savefig(join(FOLDER, f"ScatterPlot.png"), dpi=600)
    plt.close("all")
    mse = mean_squared_error(y_test,  y_predict)
    print(f"The error is {mse:.2f}%")

    results = {"n_params": n_params,
               "split_val": split_val,
               "param_date": param_date,
               "mse": mse}

    return results


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data_for_training()
    y_test, y_predict = get_prediction(X_train, X_test, y_train, y_test)
    results = plot_prediction(y_test, y_predict)
