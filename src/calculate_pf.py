from pandapower.timeseries.data_sources.frame_data import DFData
from pandapower.control.controller.const_control import ConstControl
import pandapower.timeseries as ts
import matplotlib.pyplot as plt
import os
import sys
from os.path import join

import pandapower as pp
import pandas as pd
import seaborn as sns
sns.set_style("darkgrid")

# Adding path for common functions
path = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(path)
from src.common import DATA_DIR, DEV_GRID_RESULTS_DIR, FIG_RESULTS_DIR, RUN_GRID_RESULTS_DIR
from src.config import STATUS

def plot_power_data():
    household_loads_raw = pd.read_csv(join(DATA_DIR,"household_loads.csv"),
                                parse_dates=["Time"], skiprows=1)
    pv_production_raw = pd.read_csv(join(DATA_DIR,"pv_production.csv"),
                                parse_dates=["Time"])
    for col in pv_production_raw.columns:
        if "Time" not in col:
            pv_production_raw[col]*=1/2

    if "Time" in household_loads_raw.columns:
        household_loads_raw.set_index("Time", inplace=True)
    sum_hh =  household_loads_raw.sum(axis=1)
    sum_hh*=1/1000
    if len(sum_hh)>4*24*14:
        sum_hh_winter=sum_hh[:4*24*14]
    
    if len(sum_hh)>4*24*194:
        sum_hh_summer = sum_hh[4*24*180:4*24*194]


    if "Time" in pv_production_raw.columns:
        pv_production_raw.set_index("Time", inplace=True)

    sum_pv= pv_production_raw.sum(axis=1)
    sum_pv*=1/1000
    if len(sum_pv)>4*24*14:
        sum_pv_winter=sum_pv[:4*24*14]
    if len(sum_pv)>4*24*194:
        sum_pv_summer = sum_pv[4*24*180:4*24*194]

    fig,ax = plt.subplots(figsize=(10,5))
    sum_hh_winter.plot(label="Total Loads", ax=ax)
    sum_pv_winter.plot(label="Total PV Generation",ax=ax, style="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Power [kW]")
    ax.set_title("Sum of Power Flows - Winter", fontweight="bold")
    fig.savefig(join(FIG_RESULTS_DIR, "power_data_winter.png"), dpi=600)
    
    if len(sum_pv)>4*24*194:
        fig,ax = plt.subplots(figsize=(10,5))
        sum_hh_summer.plot(label="Total Loads", ax=ax)
        sum_pv_summer.plot(label="Total PV Generation",ax=ax, style="--")
        ax.set_xlabel("Date")
        ax.set_ylabel("Power [kW]")
        ax.set_title("Sum of Power Flows - Summer", fontweight="bold")
        fig.savefig(join(FIG_RESULTS_DIR, "power_data_summer.png"), dpi=600)

def get_power_data():
    household_loads_raw = pd.read_csv(join(DATA_DIR,"household_loads.csv"),
                                parse_dates=True, skiprows=1)
    meteodata_raw = pd.read_csv(join(DATA_DIR,"meteodata.csv"),
                            parse_dates=True, skiprows=1)
    pv_production_raw = pd.read_csv(join(DATA_DIR,"pv_production.csv"),
                                parse_dates=True)
    for col in pv_production_raw.columns:
        if "Time" not in col:
            pv_production_raw[col]*=1/2

    household_loads= household_loads_raw.drop(["Time"],axis=1)
    meteodata=meteodata_raw.drop(["Time"],axis=1)
    pv_production = pv_production_raw.drop(["Time"],axis=1)
    range_bus = range(len(household_loads.keys()))
    household_loads.rename(columns={f"House{i}":f"house_{i}" for i in range_bus}, inplace = True)
    pv_production.rename(columns={f"PV{i}":f"pv_{i}" for i in range_bus}, inplace = True)

    # Trim the data while I work on the rest of the functions
    if STATUS == "develop":
        household_loads = household_loads.head(100)
        meteodata = meteodata.head(100)
        pv_production = pv_production.head(100)


    return household_loads, meteodata, pv_production

def run_power_flow(household_loads,pv_production):
    range_bus = len(household_loads.keys())

    pp_file = join(DATA_DIR, "grid_model.json")
    net = pp.from_json(filename = pp_file)

    ds_sgen = DFData(pv_production)
    ConstControl(net, "sgen", "p_mw", element_index=net.sgen.index[0:range_bus],
                profile_name=pv_production.columns, data_source=ds_sgen)
    ds_load = DFData(household_loads)
    ConstControl(net, "load", "p_mw", element_index=net.load.index[0:range_bus],
                profile_name=household_loads.columns, data_source=ds_load)

    output_path= DEV_GRID_RESULTS_DIR
    if STATUS == "run":
        output_path =RUN_GRID_RESULTS_DIR

    ts.OutputWriter(net, output_path=output_path, output_file_type=".json")
    ts.run_timeseries(net)



if __name__ == "__main__":

    plot_power_data()
    household_loads, meteodata, pv_production =  get_power_data()
    run_power_flow(household_loads,pv_production)
    pass
