import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycontrails import Flight, Fleet
from pycontrails.core import datalib, models
from pycontrails.physics import units
from pycontrails.models.cocip import Cocip
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.models.ps_model import PSFlight
from pycontrails.models import emissions
from pycontrails.models.dry_advection import DryAdvection
from datetime import datetime, timedelta

def main():
    # define sim properties
    sim_time = ("2022-01-20 12:00:00", "2022-03-20 12:00:00")
    bbox = (-0.5, 2.5, -0.5, 2.5, 11000, 12000)
    horiz_res = 1 # degrees lat/lon
    vert_res = 100 # meters
    ts_met = "6H" # met data time step
    ts_traj = "1min" # trajectory time step
    ts_chem = "20s" # chemistry time step

    flight_time = ("2022-03-01 00:00:00", "2022-03-01 06:00:00") # how long plumes are simulated for as this is the only met data i have atm.
    pressure_levels = (400, 300, 200, 100)

    era5pl = ERA5(
        time=flight_time,
        variables=["t", "q", "u", "v", "w", "z", "r"],
        pressure_levels=pressure_levels,
    )
    #era5sl = ERA5(time=time_bounds, variables=Cocip.rad_variables)

    met = era5pl.open_metdataset()
    #rad = era5sl.open_metdataset()

    # define start coords
    coords0 = (0, 0, 11500) # lon, lat, alt
    speed = 100 # m/s
    theta = 45 # degrees
     # lon_min, lon_max, lat_min, lat_max, alt_min, alt_max

    fl = {}
    dry_adv_df = {}

    # generate trajectory
    fl[0] = traj_gen(coords0, flight_time, ts_traj, speed, theta, bbox)
    fl[0].attrs = {"flight_id": int(0), "aircraft_type": "A320"}
    fl[0]["air_temperature"] = models.interpolate_met(met, fl[0], "air_temperature")
    ax = plt.axes()
    ax.set_xlim([bbox[2], bbox[3]])
    ax.set_ylim([bbox[0], bbox[1]])
    ax.set_aspect(1)


    for i in range(1, 10):
        fl[i] = follow_traj(coords0, flight_time, ts_traj, speed, theta, bbox, 5000*i, 1000*i, 0)
        fl[i].attrs = {"flight_id": int(i), "aircraft_type": "A320"}
        fl[i]["air_temperature"] = models.interpolate_met(met, fl[i], "air_temperature")


        dt_integration = pd.Timedelta(minutes=10)
        max_age = pd.Timedelta(hours=6)

        params = {
            "dt_integration": dt_integration,
            "max_age": max_age,
            "depth": 50.0,  # initial plume depth, [m]
            "width": 40.0,  # initial plume width, [m]
        }
        print(fl[i].dataframe)

        dry_adv = DryAdvection(met, params)
        dry_adv
        dry_adv_df[i] = dry_adv.eval(fl[i]).dataframe  
        dry_adv_df[i].to_csv("/home/ktait98/home_repo/2024/pycontrails_runs/dry_adv" + repr(i) + ".csv")

        ax.scatter(
            fl[i]["longitude"], fl[i]["latitude"], s=3, color="red", label="Flight path"
        )

        ax.scatter(
            dry_adv_df[i]["longitude"], dry_adv_df[i]["latitude"], s=0.1, label="Plume evolution"
        )

        ax.legend()
        ax.set_title("Flight path and plume evolution under dry advection")

    # calc ac performance for each fl instance
    # for i in range(1, 10):

    #     fl[i] = follow_traj(coords0, flight_time, ts_traj, speed, theta, bbox, 2000*i, 1000*i, 0)

    #     fl[i].attrs = {"flight_id": int(i), "aircraft_type": "A320"}

    #     # calc air temp using ISA
    #     fl[i]["air_temperature"] = units.m_to_T_isa(fl[i]["altitude"])

    #     # estimate airspeed using groundspeed
    #     fl[i]["true_airspeed"] = fl[i].segment_groundspeed()
 
    #     # create PSFlight model and eval
    #     ps_model = PSFlight()
    #     fl[i] = ps_model.eval(fl[i])

    #     print(fl[i].dataframe)
    #     fl[i].plot(ax=ax)

    
    plt.show()


def traj_gen(coords, flight_time, ts_traj, speed, theta, bbox):

    # Convert heading angle to radians
    theta_rad = np.radians(theta)

    # Calculate time array
    time = np.array(datalib.parse_timesteps(flight_time, freq=ts_traj), dtype="datetime64[s]")
    dt = (time - time[0]).astype('timedelta64[s]').astype(int)

    # Calculate trajectory points
    lats = coords[0] + ((speed * dt * np.cos(theta_rad))/6378E+03) * (180/np.pi)
    lons = coords[1] + ((speed * dt * np.sin(theta_rad))/6378E+03) * (180/np.pi) / np.cos(lats * np.pi/180)
    alts = coords[2] + np.zeros_like(time.astype(int))  # For simplicity, assume straight-line 
    # Filter points within bounding box
    mask = (
        (lats >= bbox[0]) & (lats <= bbox[1]) &
        (lons >= bbox[2]) & (lons <= bbox[3]) &
        (alts >= bbox[4]) & (alts <= bbox[5])
    )
   
    fl = Flight(longitude=lons[mask], latitude=lats[mask], altitude=alts[mask], time=time[mask])

    return fl

def follow_traj(coords, flight_time, ts_traj, speed, theta, bbox, dx, dy, dz):

    theta_rad = np.radians(theta)
   
    # Calculate time array
    time = np.array(datalib.parse_timesteps(flight_time, freq=ts_traj), dtype="datetime64[s]")
    dt = (time - time[0]).astype('timedelta64[s]').astype(int)

    lat0 = coords[0] + ((dx*np.cos(theta_rad) - dy*np.sin(theta_rad))/6378E+03) * (180/np.pi)
    lon0 = coords[1] + ((dx*np.sin(theta_rad) + dy*np.cos(theta_rad))/6378E+03) * (180/np.pi) / np.cos((coords[0]) * np.pi/180)
    alt0 = coords[2] + dz

    lats = lat0 + ((speed * dt * np.cos(theta_rad))/6378E+03) * (180/np.pi)
    lons = lon0 + ((speed * dt * np.sin(theta_rad))/6378E+03) * (180/np.pi) / np.cos(lats * np.pi/180)
    alts = alt0 + np.zeros_like(time.astype(int))  # For simplicity, assume straight-line 

    # Filter points within bounding box
    mask = (
        (lats >= bbox[0]) & (lats <= bbox[1]) &
        (lons >= bbox[2]) & (lons <= bbox[3]) &
        (alts >= bbox[4]) & (alts <= bbox[5])
    )

    fl = Flight(longitude=lons[mask], latitude=lats[mask], altitude=alts[mask], time=time[mask], flight_id=1)

    return fl



def plot_trajectory(ax, x, y, z, label):
    ax.plot3D(x, y, z, label=label)


# plot_trajectory(ax, x2, y2, z2, "Flight 2")
# ax.set_xlabel('X (meters)')
# ax.set_ylabel('Y (meters)')
# ax.set_zlabel('Altitude (meters)')
# ax.legend()

# plt.show()


# def ac_performance(traj, ac_type, met):
#     return

# def emissions()
    
if __name__=="__main__":
    main()