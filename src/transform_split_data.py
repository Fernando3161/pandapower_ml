
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from os.path import join
import sys
import os

path = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(path)
from src.common import DEV_GRID_RESULTS_DIR, DATA_DIR, RUN_GRID_RESULTS_DIR
from src.config import STATUS

def get_data_for_training(config = None):
    n_params =6
    split_val=0.8
    param_date=True

    if config:
        n_params =config["n_params"] 
        split_val=config["split_val"]
        param_date=config["param_date"]

    meteodata_raw = pd.read_csv(join(DATA_DIR, "meteodata.csv"),
                                parse_dates=["Time"], skiprows=1)

    x = meteodata_raw
    y = None
    if STATUS == "develop":
        y = pd.read_json(
            join(DEV_GRID_RESULTS_DIR, "res_line", "loading_percent.json"))
    
    if STATUS == "run":
        y = pd.read_json(
            join(RUN_GRID_RESULTS_DIR, "res_line", "loading_percent.json"))

    x = x[['Time',
          'AirTemperature',
           'RelHumidity',
           'DewPoint',
           'WetBulbTempHourly',
           'GlobalRadiation',
           'WindSpeed',
           'DiffRadiation',
           'DirectNormalRad',
           'CloudCoverage0',
           'AirPressHourly',
           ]]

    if n_params not in range(6, len(x.columns)+1):
        raise ValueError(f"n_params must be between 6 and {len(x.columns)}")

    columns = x.columns[0:n_params]
    x=x[columns]
    
    if param_date is True:
        x["month"] = [y.month for y in x["Time"]]
        x["day"] = [y.day for y in x["Time"]]
        x["hour"] = [y.hour for y in x["Time"]]
        x["minute"] = [y.minute for y in x["Time"]]

    x_data = x.drop(["Time"], axis=1)

    if STATUS == "develop":
        x_data = x_data.head(100)

    if split_val > 0.95 or split_val < 0.5:
        raise ValueError(f"split_val must be between 0.5 and 0.95")

    X_train, X_test, y_train, y_test = train_test_split(
        x_data, y, train_size=split_val)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
