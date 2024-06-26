import numpy as np
import pandas as pd
import xarray as xr
import pathlib
import numpy.typing as npt


import math
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import Image, display
import seaborn as sns
from pyproj import Geod
from pycontrails import Flight, Fleet, GeoVectorDataset
from pycontrails.core.met import MetDataset
from pycontrails.core import datalib, models
from pycontrails.physics import units
from pycontrails.models.cocip import Cocip, contrails_to_hi_res_grid
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.models.ps_model import PSFlight
from pycontrails.models import emissions
from pycontrails.models.dry_advection2 import DryAdvection
from datetime import datetime, timedelta

pd.set_option('display.max_rows', None)

def main():
    # define sim properties
    sim_time = ("2022-03-01 00:00:00", "2022-03-01 01:00:00")
    flight_time = ("2022-03-01 00:00:00", "2022-03-01 01:00:00")
    bbox = (-1, -1, 0, 0, 11000, 12000) # lon_min, lat_min, lon_max, lat_max, alt_min, alt_max
    levels = [400, 300, 200, 100] # pressure levels
    horiz_res = 0.003 # degrees lat/lon
    vert_res = 100 # meters
    ts_met = "6H" # met data time step
    ts_traj = "5min" # trajectory time step
    ts_chem = "20s" # chemistry time step

    # define start coords
    coords0 = (-.5, -.5, 11500) # lon, lat, alt
    speed = 100 # m/s
    theta = 0 # degrees
    sep_long = 1000 # m
    sep_lat = 5000 # m
    sep_alt = 0 # m
    n_ac = 1

    longitudes = np.arange(bbox[0]+0.1, bbox[2]-0.1, horiz_res)
    latitudes = np.arange(bbox[1]+0.1, bbox[3]-0.1, horiz_res)
    altitudes = np.arange(bbox[4]+vert_res, bbox[5]-vert_res, vert_res)
    timesteps = pd.date_range(sim_time[0], sim_time[1], freq=ts_traj)
    

    met = generate_met(
        longitudes=longitudes, latitudes=latitudes, levels=levels, timesteps=timesteps,
        air_temperature=240.0, specific_humidity=0.00012,
        u_component_of_wind=0.0, u_component_of_wind_lower=-40.0,
        v_component_of_wind=0.0, v_component_of_wind_lower=-0.0,
        vertical_velocity=0.0,
        geopotential=10000.0,
        relative_humidity=0.5,
    )

    print(met)

    # init fl and dry_adv dicts
    fl_dict = {}
    dry_adv_dict = {}
    emi_dict = {}

    for i in range(1, n_ac + 1):
        # generate flight paths
        fl_dict = traj_gen(i, fl_dict, met, coords0, flight_time, ts_traj, speed, theta, bbox, horiz_res, sep_long, sep_lat, sep_alt)

        # do dry advection
        dry_adv_dict = dry_advection(i, dry_adv_dict, fl_dict, met)

        # convert fl_dict and dry_adv_dict to dataframes
        fl_dict[i] = fl_dict[i].dataframe
        dry_adv_dict[i] = dry_adv_dict[i].dataframe
        
        # calculate headings
        fl_dict[i] = calculate_heading_fl(fl_dict[i])
        dry_adv_dict[i] = calculate_heading_pl(dry_adv_dict[i])
        
        # drop emissions totals (per m is key here) and add waypoint column
        fl_dict[i].drop(columns=["co2", "nox", "h2o", "so2", "sulphates", "co", "hc", "nvpm_mass", "nvpm_number"], inplace=True)
        dry_adv_dict[i]["flight_id"] = fl_dict[i]["flight_id"][0]
        #print(fl_dict[i].head(500))
        #print(dry_adv_dict[i].sort_values(by=['time', 'waypoint']).head(500))

        fl_dict[i]["waypoint"] = fl_dict[i].index

    # ANIMATE PLOT

    # concatenate all dfs into one
    fl_df = pd.concat(fl_dict.values())
    dry_adv_df = pd.concat(dry_adv_dict.values())

    fig1, ax1 = plt.subplots()

    scat_fl = ax1.scatter([], [], s=5, c="red", label="Flight path")
    scat_pl = ax1.scatter([], [], s=0.1, c="blue", label="Plume evolution")
    ax1.legend(loc="upper left")
    ax1.set_xlim([bbox[0], bbox[2]])
    ax1.set_ylim([bbox[1], bbox[3]])
    
    dt_integration = pd.Timedelta(minutes=5)
    source_frames = fl_df.groupby(fl_df["time"].dt.ceil(dt_integration))
    dry_adv_frames = dry_adv_df.groupby(dry_adv_df["time"].dt.ceil(dt_integration))

    times = dry_adv_frames.indices

    def animate(t):
        ax1.set_title(t)

        try:
            group = source_frames.get_group(t)
        except KeyError:
            offsets = [[None, None]]
        else:
            offsets = group[["longitude", "latitude"]]
        
        scat_fl.set_offsets(offsets)

        group = dry_adv_frames.get_group(t)
        offsets = group[["longitude", "latitude"]]
        width = 10E-3 * group["width"]
        scat_pl.set_offsets(offsets)
        scat_pl.set_sizes(width)

        return scat_fl, scat_pl
    
    plt.close()
    ani = FuncAnimation(fig1, animate, frames=times)
    filename = pathlib.Path("evo.gif")
    ani.save(filename, dpi=300, writer=PillowWriter(fps=10))
    
    plume_df = pd.merge(
                fl_df[['flight_id', 'waypoint', 'fuel_flow', 'true_airspeed', 'co2_m', 'h2o_m',
                'so2_m', 'nox_m', 'co_m', 'hc_m', 'nvpm_m']],
                dry_adv_df[['flight_id', 'waypoint', 'time', 'longitude', 'latitude', 'level', 'width', 'heading']], on=['flight_id', 'waypoint']).sort_values(by=['time', 'flight_id', 'waypoint'])
        
    plume_df['sin_a'] = np.sin(np.radians(plume_df['heading']))
    plume_df['cos_a'] = np.cos(np.radians(plume_df['heading']))
    
    plume_df['altitude'] = plume_df['level']

    print(plume_df.sort_values(by=['flight_id', 'time', 'waypoint']).head(500))

    spatial_bbox = (bbox[0], bbox[1], bbox[2], bbox[3])
    plume_to_grid(flight_time, spatial_bbox, horiz_res, plume_df)
    
    
# METEOROLOGY
def generate_met(
        longitudes: npt.NDArray[np.float_],
        latitudes: npt.NDArray[np.float_],
        levels: npt.NDArray[np.float_],
        timesteps: pd.DatetimeIndex,
        air_temperature: float,
        specific_humidity: float,
        u_component_of_wind: float,
        u_component_of_wind_lower: float,
        v_component_of_wind: float,
        v_component_of_wind_lower: float,
        vertical_velocity: float,
        geopotential: float,
        relative_humidity: float,
    ) -> MetDataset:


        # create wind fields
        shape = (len(longitudes), len(latitudes), len(levels), len(timesteps))
        u_wind_field = np.zeros(shape)
        v_wind_field = np.zeros(shape)

        for l in range(len(levels)):
            for t in range(len(timesteps)):
                u_wind_field[:, :, l, t] = np.full((len(longitudes), len(latitudes)), u_component_of_wind - l/len(levels)*(u_component_of_wind - u_component_of_wind_lower))
                
                v_wind_field[:, :, l, t] = np.full((len(longitudes), len(latitudes)), v_component_of_wind - l/len(levels)*(v_component_of_wind - v_component_of_wind_lower))

        # create artificial metdataset
        met = xr.Dataset(
            {
                "air_temperature": (["longitude", "latitude", "level", "time"], np.full((len(longitudes), len(latitudes), len(levels), len(timesteps)), air_temperature)),
                "specific_humidity": (["longitude", "latitude", "level", "time"], np.full((len(longitudes), len(latitudes), len(levels), len(timesteps)), specific_humidity)),
                "eastward_wind": (["longitude", "latitude", "level", "time"], u_wind_field),
                "northward_wind": (["longitude", "latitude", "level", "time"], v_wind_field),
                "lagrangian_tendency_of_air_pressure": (["longitude", "latitude", "level", "time"], np.full((len(longitudes), len(latitudes), len(levels), len(timesteps)), vertical_velocity)),
                "geopotential": (["longitude", "latitude", "level", "time"], np.full((len(longitudes), len(latitudes), len(levels), len(timesteps)), geopotential)),
                "relative_humidity": (["longitude", "latitude", "level", "time"], np.full((len(longitudes), len(latitudes), len(levels), len(timesteps)), relative_humidity)),
            },
            coords={
                "longitude": longitudes,
                "latitude": latitudes,
                "level": levels,
                "time": timesteps,
            },
        )

        return MetDataset(met)

# TRAJ GEN
def traj_gen(i, fl_dict, met, coords, flight_time, ts_traj, speed, theta, bbox, horiz_res, sep_long, sep_lat, sep_alt):

    if i == 0:
        # generate leader trajectory
        fl_dict[i] = leader_traj(coords, flight_time, ts_traj, speed, theta, bbox, horiz_res)
        fl_dict[i].attrs = {"flight_id": int(0), "aircraft_type": "A320"}
        fl_dict[i]["air_temperature"] = models.interpolate_met(met, fl_dict[i], "air_temperature")
        fl_dict[i]["specific_humidity"] = models.interpolate_met(met, fl_dict[i], "specific_humidity")

    else:
        # generate follower flights
        fl_dict[i] = follow_traj(coords, flight_time, ts_traj, speed, theta, bbox, horiz_res, sep_long*i, sep_lat*i, sep_alt*i)
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

def leader_traj(coords, flight_time, ts_traj, speed, theta, bbox, horiz_res):

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
        (lats > bbox[0]-horiz_res) & (lats < bbox[2]+horiz_res) &
        (lons > bbox[1]-horiz_res) & (lons < bbox[3]+horiz_res) &
        (alts > bbox[4]) & (alts < bbox[5])
    )
   
    fl = Flight(longitude=lons[mask], latitude=lats[mask], altitude=alts[mask], time=time[mask])

    return fl

def follow_traj(coords, flight_time, ts_traj, speed, theta, bbox, horiz_res, sep_long, sep_lat, sep_alt):

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
        (lats > bbox[0]+horiz_res) & (lats < bbox[2]-horiz_res) &
        (lons > bbox[1]+horiz_res) & (lons < bbox[3]-horiz_res) &
        (alts > bbox[4]) & (alts < bbox[5])
    )

    fl = Flight(longitude=lons[mask], latitude=lats[mask], altitude=alts[mask], time=time[mask], flight_id=1)

    return fl

# DRY ADVECTION
def dry_advection(i, dry_adv_dict, fl_dict, met):
            
    # do dry advection
    dt_integration = pd.Timedelta(minutes=5)
    max_age = pd.Timedelta(hours=2)

    params = {
        "dt_integration": dt_integration,
        "max_age": max_age,
        "depth": 50.0,  # initial plume depth, [m]
        "width": 40.0,  # initial plume width, [m]
    }

    dry_adv = DryAdvection(met, params)
    dry_adv

    dry_adv_dict[i] = dry_adv.eval(fl_dict[i])

    dry_adv_dict[i] = calculate_advection_distance(dry_adv_dict[i])

    return dry_adv_dict

def calculate_advection_distance(dry_adv_df):
    # Original coordinates
    orig_lon = np.radians(dry_adv_df["longitude"][0])
    orig_lat = np.radians(dry_adv_df["latitude"][0])
    new_lon = np.radians(dry_adv_df["longitude"])
    new_lat = np.radians(dry_adv_df["latitude"])

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

    dry_adv_df["advection_distance"] = distance

    return dry_adv_df

def calculate_heading_fl(fl_df):
    g = Geod(ellps="WGS84")

    startlat = fl_df['latitude'].values[:-1]
    startlon = fl_df['longitude'].values[:-1]
    endlat = fl_df['latitude'].values[1:]
    endlon = fl_df['longitude'].values[1:]
    az12, az21, dist = g.inv(startlon, startlat, endlon, endlat)

    fl_df["heading"] = np.concatenate([[np.nan], az12])

    return fl_df

def calculate_heading_pl(dry_adv_df):
    # Sort the dataframe by time and waypoint
    dry_adv_df = dry_adv_df.sort_values(by=["time", "waypoint"])

    # Group the dataframe by the timestep and apply the function
    dry_adv_df['heading'] = dry_adv_df.groupby('time').apply(calculate_heading_g).reset_index(drop=True)

    return dry_adv_df

def calculate_heading_g(group):
    g = Geod(ellps="WGS84")

    startlat = group['latitude'].values[:-1]
    startlon = group['longitude'].values[:-1]
    endlat = group['latitude'].values[1:]
    endlon = group['longitude'].values[1:]
    az12, az21, dist = g.inv(startlon, startlat, endlon, endlat)

    heading = (90 - az12) % 360
    
    return pd.Series(np.concatenate([[np.nan], heading]), index=group.index)

def calc_continuous(plume: GeoVectorDataset):
    """Calculate the continuous segments of this timestep.

    Mutates parameter ``contrail`` in place by setting or updating the
    "continuous" variable.

    Parameters
    ----------
    contrail : GeoVectorDataset
        GeoVectorDataset instance onto which "continuous" is set.

    Raises
    ------
    ValueError
        If ``contrail`` is empty.
    """
    if not plume:
        raise ValueError("Cannot calculate continuous on an empty contrail")
    same_flight = plume["flight_id"][:-1] == plume["flight_id"][1:]
    consecutive_waypoint = np.diff(plume["waypoint"]) == 1
    continuous = np.empty(plume.size, dtype=bool)
    continuous[:-1] = same_flight & consecutive_waypoint
    continuous[-1] = False  # This fails if contrail is empty
    plume.update(continuous=continuous)  # overwrite continuous

# PLUME TO GRID
def plume_to_grid(flight_time, bbox, horiz_res, plume_df):

    # loop over time and plume property
    plume_data = []
    for t, time in enumerate(plume_df['time'].unique()):
        plume_time_data = []
        for p in ['h2o_m']: #, 'co2_m', 'so2_m', 'nox_m', 'co_m', 'hc_m', 'nvpm_m']:
            
            # create geovectordataset to store instantaneous plume data
            plume_time_data = GeoVectorDataset(data=plume_df.loc[plume_df['time'] == time])

            calc_continuous(plume_time_data)
            # call contrails_to_hi_res_grid
            plume_time_data = contrails_to_hi_res_grid(time=time, 
                                                contrails_t=plume_time_data,
                                                var_name=p, 
                                                spatial_bbox=bbox, spatial_grid_res=horiz_res)
        plume_data.append(plume_time_data)
        np.savetxt("plumes/plume_data_" + repr(t) + ".csv", plume_time_data, delimiter=",")    
    plume_data = np.array(plume_data)
    

    # plume_frames = plume_df.groupby(plume_df["time"].dt.ceil(pd.Timedelta(seconds=30)))

    # times = plume_frames.indices
    

    def heatmap_func(i):
        ax.cla()
        sns.heatmap(plume_data[i, ...].T,
        ax = ax,
        cbar = True,
        cbar_ax = cbar_ax,
        vmin = plume_data.min(),
        vmax = 0.002,
        )
        ax.invert_yaxis()

    grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
    fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw = grid_kws, figsize = (12, 8))  

    anim = FuncAnimation(fig = fig, func = heatmap_func, frames = 80, blit = False)
    
    print(plume_data)

    filename = pathlib.Path("plume.gif")
    #plt.show()
    anim.save(filename, dpi=300, writer=PillowWriter(fps=10))

if __name__=="__main__":
    main()