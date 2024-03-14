import numpy as np
import pandas as pd
import math
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycontrails import Flight, Fleet, GeoVectorDataset
from pycontrails.core import datalib, models
from pycontrails.physics import units
from pycontrails.models.cocip import Cocip
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.models.ps_model import PSFlight
from pycontrails.models import emissions
from pycontrails.models.dry_advection2 import DryAdvection
from datetime import datetime, timedelta

def main():
    # define sim properties
    sim_time = ("2022-01-20 12:00:00", "2022-03-20 12:00:00")
    bbox = (-0.5, 2.5, -0.5, 2.5, 11000, 12000) # lon_min, lon_max, lat_min, lat_max, alt_min, alt_max
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

    met = era5pl.open_metdataset()
    #rad = era5sl.open_metdataset()

    print(met['eastward_wind'].values)
    print(met['northward_wind'].values)    
    

    # define start coords
    coords0 = (0, 0, 11500) # lon, lat, alt
    speed = 100 # m/s
    theta = 45 # degrees
    sep_long = 5000 # m
    sep_lat = 1000 # m
    sep_alt = 0 # m
    n_ac = 10

    # init fl and dry_adv dicts
    fl_dict = {}
    dry_adv_dict = {}
    concs_dict = {}

    # set up flight path plotting
    ax = plt.axes()
    ax.set_xlim([bbox[2], bbox[3]])
    ax.set_ylim([bbox[0], bbox[1]])
    ax.set_aspect(1)


    for i in range(0, n_ac):
        # generate flight paths
        fl_dict = traj_gen(i, fl_dict, met, coords0, flight_time, ts_traj, speed, theta, bbox, sep_long, sep_lat, sep_alt)

        dry_adv_dict = dry_advection(i, dry_adv_dict, fl_dict, met)

        # convert fl_dict and dry_adv_dict to dataframes
        fl_dict[i] = fl_dict[i].dataframe
        dry_adv_dict[i] = dry_adv_dict[i].dataframe

        fl_dict[i].drop(columns=["co2", "nox", "h2o", "so2", "sulphates", "co", "hc", "nvpm_mass", "nvpm_number"], inplace=True)
        fl_dict[i]["waypoint"] = fl_dict[i].index
        #fl_dict[i]["time"] = pd.to_datetime(fl_dict[i]["time"])     

        # # merge fl_dict and dry_adv_dict
        # concs_dict[i] = pd.merge(
        #     fl_dict[i][["time", "waypoint", "flight_id", "longitude", "latitude", "altitude", , ]],
        # )

        ax.plot(fl_dict[i]["longitude"], fl_dict[i]["latitude"], color="red", label="Flight path")
        ax.scatter(dry_adv_dict[i]["longitude"], dry_adv_dict[i]["latitude"], s=0.1, label="Plume evolution")

    ax.legend()
    ax.set_title("Flight path and plume evolution under dry advection")


    # Plotting width and depth of each plume waypoint against time
    for i in range(1):
        plt.figure()
        
        plt.scatter(dry_adv_dict[i]["time"], dry_adv_dict[i]["advection_distance"], label="Advection Distance")
        plt.scatter(dry_adv_dict[i]["time"], dry_adv_dict[i]["width"], label="Width")
        plt.scatter(dry_adv_dict[i]["time"], dry_adv_dict[i]["depth"], label="Depth")
        plt.xlabel("Time")
        plt.ylabel("Width/Depth")
        plt.title(f"Plume Waypoint {i+1}")
        plt.legend()
        plt.show()

    (fl_dict[0]).to_csv("fl.csv")
    (dry_adv_dict[0]).to_csv("plume_evolution.csv")


def traj_gen(i, fl_dict, met, coords, flight_time, ts_traj, speed, theta, bbox, sep_long, sep_lat, sep_alt):

    if i == 0:
        # generate leader trajectory
        fl_dict[0] = leader_traj(coords, flight_time, ts_traj, speed, theta, bbox)
        fl_dict[0].attrs = {"flight_id": int(0), "aircraft_type": "A320"}
        fl_dict[0]["air_temperature"] = models.interpolate_met(met, fl_dict[0], "air_temperature")
        fl_dict[i]["specific_humidity"] = models.interpolate_met(met, fl_dict[i], "specific_humidity")

    else:
        # generate follower flights
        fl_dict[i] = follow_traj(coords, flight_time, ts_traj, speed, theta, bbox, sep_long*i, sep_lat*i, sep_alt*i)
        fl_dict[i].attrs = {"flight_id": int(i), "aircraft_type": "A320"}
        fl_dict[i]["air_temperature"] = models.interpolate_met(met, fl_dict[i], "air_temperature")
        fl_dict[i]["specific_humidity"] = models.interpolate_met(met, fl_dict[i], "specific_humidity")

    # get ac_performance
    fl_dict[i]["true_airspeed"] = fl_dict[i].segment_groundspeed()

    ps_model = PSFlight()
    fl_dict[i] = ps_model.eval(fl_dict[i])

    # get emissions
    emi = emissions.Emissions()
    fl_dict[i] = emi.eval(fl_dict[i])

    # get em mass per metre
    fl_dict[i]["co2_m"] = 3.16 * fl_dict[i]["fuel_flow"] / fl_dict[i]["true_airspeed"]
    fl_dict[i]["h2o_m"] = 1.23 * fl_dict[i]["fuel_flow"] / fl_dict[i]["true_airspeed"]
    fl_dict[i]["so2_m"] = 0.00084 * fl_dict[i]["fuel_flow"] / fl_dict[i]["true_airspeed"]
    fl_dict[i]["nox_m"] = fl_dict[i]["nox_ei"] * fl_dict[i]["fuel_flow"] / fl_dict[i]["true_airspeed"]
    fl_dict[i]["co_m"] = fl_dict[i]["co_ei"] * fl_dict[i]["fuel_flow"] / fl_dict[i]["true_airspeed"]
    fl_dict[i]["hc_m"] = fl_dict[i]["hc_ei"] * fl_dict[i]["fuel_flow"] / fl_dict[i]["true_airspeed"]
    fl_dict[i]["nvpm_m"] = fl_dict[i]["nvpm_ei_m"] * fl_dict[i]["fuel_flow"] / fl_dict[i]["true_airspeed"]

    return fl_dict
        
def dry_advection(i, dry_adv_dict, fl_dict, met):
            
    # do dry advection
    dt_integration = pd.Timedelta(minutes=10)
    max_age = pd.Timedelta(hours=6)

    params = {
        "dt_integration": dt_integration,
        "max_age": max_age,
        "depth": 50.0,  # initial plume depth, [m]
        "width": 40.0,  # initial plume width, [m]
    }

    dry_adv = DryAdvection(met, params)
    dry_adv

    dry_adv_dict[i] = dry_adv.eval(fl_dict[i])

    dry_adv_dict[i] = calculate_max_width_coords(dry_adv_dict[i])

    dry_adv_dict[i] = calculate_advection_distance(dry_adv_dict[i])

    dry_adv_dict[i] = calculate_azimuth(dry_adv_dict[i])

    return dry_adv_dict

def leader_traj(coords, flight_time, ts_traj, speed, theta, bbox):

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

def follow_traj(coords, flight_time, ts_traj, speed, theta, bbox, sep_long, sep_lat, sep_alt):

    theta_rad = np.radians(theta)
   
    # Calculate time array
    time = np.array(datalib.parse_timesteps(flight_time, freq=ts_traj), dtype="datetime64[s]")
    dt = (time - time[0]).astype('timedelta64[s]').astype(int)

    lat0 = coords[0] + ((sep_long*np.cos(theta_rad) - sep_lat*np.sin(theta_rad))/6378E+03) * (180/np.pi)
    lon0 = coords[1] + ((sep_long*np.sin(theta_rad) + sep_lat*np.cos(theta_rad))/6378E+03) * (180/np.pi) / np.cos((coords[0]) * np.pi/180)
    alt0 = coords[2] + sep_alt

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

def calculate_max_width_coords(dry_adv_dict):
    # Convert azimuth to radians
    azimuth_rad = np.radians(dry_adv_dict["azimuth"])

    # Earth's radius in meters (approximate value, can be adjusted based on specific requirements)
    earth_radius = 6371000.0

    # Convert width from meters to degrees
    width_deg = dry_adv_dict["width"] / (earth_radius * np.cos(np.radians(dry_adv_dict["latitude"])))


    # Calculate coordinates of points at ±w/2 from the center
    dry_adv_dict["lon_plus"] = dry_adv_dict["longitude"] + (width_deg / 2) * np.cos(azimuth_rad)
    dry_adv_dict["lat_plus"] = dry_adv_dict["latitude"] + (width_deg / 2) * np.sin(azimuth_rad)

    dry_adv_dict["lon_minus"] = dry_adv_dict["longitude"] - (width_deg / 2) * np.cos(azimuth_rad)
    dry_adv_dict["lat_minus"] = dry_adv_dict["latitude"] - (width_deg / 2) * np.sin(azimuth_rad)

    return dry_adv_dict

def calculate_advection_distance(dry_adv_dict):
    # Original coordinates
    orig_lon = np.radians(dry_adv_dict["longitude"][0])
    orig_lat = np.radians(dry_adv_dict["latitude"][0])
    new_lon = np.radians(dry_adv_dict["longitude"])
    new_lat = np.radians(dry_adv_dict["latitude"])

    # Calculate the differences
    dlat = new_lat - orig_lat
    dlon = new_lon - orig_lon

    # Use the Haversine formula to calculate the distance
    a = np.sin(dlat/2)**2 + np.cos(orig_lat) * np.cos(new_lat) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    # Earth's radius in meters (approximate value, can be adjusted based on specific requirements)
    earth_radius = 6371000.0

    # Calculate the distance
    distance = earth_radius * c

    dry_adv_dict["advection_distance"] = distance

    return dry_adv_dict

if __name__=="__main__":
    main()