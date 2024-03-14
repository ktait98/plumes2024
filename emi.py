# this script combines fl and dry_adv datasets to determine emissions concentrations distributions for all waypoints and at all timesteps throughout flight

import numpy as np
import pandas as pd
import xarray as xr

fl_df = pd.read_csv("fl.csv")
print(fl_df)
plume_df = pd.read_csv("plume_evolution.csv")
print(plume_df)

concs_df = pd.merge(fl_df, plume_df, on=["time", "waypoint"])

print(concs_df)