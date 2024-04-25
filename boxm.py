import numpy as np
import pandas as pd
import dask.array as da
import xarray as xr
import pathlib
import numpy.typing as npt
import os
import time
import math
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import Image, display
from pyproj import Geod
from pycontrails import Flight, Fleet, GeoVectorDataset
from pycontrails.core.met import MetDataset
from pycontrails.core import datalib, models
from pycontrails.physics import geo, thermo, units, constants
from pycontrails.models.cocip import Cocip, contrails_to_hi_res_grid
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.models.ps_model import PSFlight
from pycontrails.models import emissions
from pycontrails.models.dry_advection2 import DryAdvection
from datetime import datetime, timedelta
from boxm_f2py import boxm_f2py
from pytimeparse.timeparse import timeparse

pd.set_option('display.max_rows', None)

def main():
    # define flight properties
    flight_time = ("2022-03-02 11:00:00", "2022-03-02 12:00:00")
    coords0 = (-.6, -.6, 11500)
    speed = 100.0 # m/s
    theta = 0.0 # degrees
    sep_long = 5000 # m
    sep_lat = 2000 # m
    sep_alt = 0 # m
    n_ac = 5
   
    dt_integration = pd.Timedelta(minutes=5)
    max_age = pd.Timedelta(hours=3)

    horiz_res_fl = 0.01 # degrees lat/lon
    vert_res_fl = 500 # meters
    ts_fl = "2min" # trajectory time step

    # define sim properties
    sim_time = ("2022-03-02 10:00:00", "2022-03-02 17:00:00")
    bbox = (-1.0, -1.0, 0.0, 0.0, 11000.0, 12000.0) # lon_min, lat_min, lon_max, lat_max, alt_min, alt_max
    levels = [400.0, 300.0, 200.0, 100.0] # pressure levels
    
    horiz_res_chem = 0.05 # degrees lat/lon
    vert_res_chem = 500 # meters    
    ts_chem = "5min" # chemistry time step
    
    lons_chem = np.arange(bbox[0] + 0.2, bbox[2] - 0.2, horiz_res_chem)
    lats_chem = np.arange(bbox[1] + 0.2, bbox[3] - 0.2, horiz_res_chem)
    alts_chem = np.arange(bbox[4] + vert_res_chem, bbox[5], vert_res_chem)
    timesteps_chem = np.array(datalib.parse_timesteps(sim_time, freq=ts_chem), dtype="datetime64[ns]")

    met = gen_met(
        lons_chem=lons_chem, lats_chem=lats_chem, levels=levels, timesteps_chem=timesteps_chem,
        air_temperature=240.0, # K
        specific_humidity=0.00012, # kg/kg
        u_component_of_wind=0.0, u_component_of_wind_lower=-0.0, # m/s
        v_component_of_wind=0.0, v_component_of_wind_lower=-0.0, # m/s
        dsn_dz=0.005, # s^-1
        vertical_velocity=0.0, # m/s
        geopotential=10000.0, # m^2/s^2
        relative_humidity=0.5, # %
    )

    bg_chem = gen_bg_chem(
         lons_chem,
         lats_chem,
         alts_chem,
         timesteps_chem
         )

    emi = gen_emi(
        met=met,
        coords0=coords0,
        flight_time=flight_time,
        ts_fl=ts_fl,
        speed=speed, # m/s
        theta=theta, # degrees
        bbox=bbox,
        horiz_res_fl=horiz_res_fl,
        vert_res_fl=vert_res_fl,
        sep_long=sep_long, # m
        sep_lat=sep_lat, # m
        sep_alt=sep_alt, # m
        dt_integration=dt_integration,
        max_age=max_age,
        n_ac=n_ac
    )

    met.data = met.data.chunk({'longitude': 'auto', 'latitude': 'auto', 'level': 'auto', 'time': 'auto'})
    emi = emi.chunk({'longitude': 'auto', 'latitude': 'auto', 'time': 'auto'})

    met.data = met.data.interp(longitude=lons_chem, latitude=lats_chem, level=units.m_to_pl(alts_chem), time=timesteps_chem, method="linear")
    emi = emi.interp(longitude=lons_chem, latitude=lats_chem, time=timesteps_chem, method="linear")
    
    #animate_pl(emi, "nox_m", 2)

    to_csvs(
         met, 
         bg_chem, 
         emi
         )
    
    run_boxm(lons_chem, lats_chem, timesteps_chem, ts_chem)
    

#################### METEOROLOGY ####################
# generate met dataset from hard-coded input (era5 still down)
def gen_met(
        lons_chem: npt.NDArray[np.float_],
        lats_chem: npt.NDArray[np.float_],
        levels: npt.NDArray[np.float_],
        timesteps_chem: pd.DatetimeIndex,
        air_temperature: float,
        specific_humidity: float,
        u_component_of_wind: float,
        u_component_of_wind_lower: float,
        v_component_of_wind: float,
        v_component_of_wind_lower: float,
        dsn_dz: float,
        vertical_velocity: float,
        geopotential: float,
        relative_humidity: float,
    ) -> MetDataset:


        # create wind fields
        shape = (len(lons_chem), len(lats_chem), len(levels), len(timesteps_chem))
        u_wind_field = np.zeros(shape)
        v_wind_field = np.zeros(shape)

        for l in range(len(levels)):
            for t in range(len(timesteps_chem)):
                u_wind_field[:, :, l, t] = np.full((len(lons_chem), len(lats_chem)), u_component_of_wind - l/len(levels)*(u_component_of_wind - u_component_of_wind_lower))
                
                v_wind_field[:, :, l, t] = np.full((len(lons_chem), len(lats_chem)), v_component_of_wind - l/len(levels)*(v_component_of_wind - v_component_of_wind_lower))

        # create artificial metdataset
        met = xr.Dataset(
            {
                "air_temperature": (["longitude", "latitude", "level", "time"], np.full((len(lons_chem), len(lats_chem), len(levels), len(timesteps_chem)), air_temperature)),
                "specific_humidity": (["longitude", "latitude", "level", "time"], np.full((len(lons_chem), len(lats_chem), len(levels), len(timesteps_chem)), specific_humidity)),
                "eastward_wind": (["longitude", "latitude", "level", "time"], u_wind_field),
                "northward_wind": (["longitude", "latitude", "level", "time"], v_wind_field),
                "lagrangian_tendency_of_air_pressure": (["longitude", "latitude", "level", "time"], np.full((len(lons_chem), len(lats_chem), len(levels), len(timesteps_chem)), vertical_velocity)),
                "wind_shear": (["longitude", "latitude", "level", "time"], np.full((len(lons_chem), len(lats_chem), len(levels), len(timesteps_chem)), dsn_dz)),
                "geopotential": (["longitude", "latitude", "level", "time"], np.full((len(lons_chem), len(lats_chem), len(levels), len(timesteps_chem)), geopotential)),
                "relative_humidity": (["longitude", "latitude", "level", "time"], np.full((len(lons_chem), len(lats_chem), len(levels), len(timesteps_chem)), relative_humidity)),
            },
            coords={
                "longitude": lons_chem,
                "latitude": lats_chem,
                "level": levels,
                "time": timesteps_chem,
            },
        )

        met = MetDataset(met)

        met = calc_M_H2O(met)

        met.data["sza"] = (('latitude', 'longitude', 'time'), calc_sza(lats_chem, lons_chem, timesteps_chem))

        return met

# calculate number density of air molecules and H2O
def calc_M_H2O(met):

        """Calculate number density of air molecules at each pressure level M"""
        N_A = 6.022e23 # Avogadro's number
        
        # Get air density from pycontrails physics.thermo script
        rho_d = met["air_pressure"].data / (constants.R_d * met["air_temperature"].data)

        # Calculate number density of air (M) to feed into box model calcs
        met.data["M"] = (N_A / constants.M_d) * rho_d * 1e-6  # [molecules / cm^3]
        met.data["M"] = met.data["M"].transpose("latitude", "longitude", "level", "time")
                
        # Calculate H2O number concentration to feed into box model calcs
        met.data["H2O"] = (met["specific_humidity"].data / constants.M_v) * N_A * rho_d * 1e-6 # [molecules / cm^3]

        # Calculate O2 and N2 number concs based on M
        met.data["O2"] = 2.079E-01 * met.data["M"]
        met.data["N2"] = 7.809E-01 * met.data["M"] 

        return met

# calculate solar zenith angles for all timesteps and locations
def calc_sza(latitudes, longitudes, timesteps):
        """Calculate szas for each cell at all timesteps"""
        sza = np.zeros((len(latitudes), len(longitudes), len(timesteps)))

        for lon, lonval in enumerate(longitudes):
                for lat, latval in enumerate(latitudes):
                        
                        theta_rad = geo.orbital_position(timesteps)
                        
                        sza[lat, lon, :] = np.arccos(geo.cosine_solar_zenith_angle(lonval, latval, timesteps, theta_rad))
        return sza

############### BACKGROUND CHEMISTRY ################
# generate bg chem dataset from species.nc
def gen_bg_chem(
        lons_chem: npt.NDArray[np.float_],
        lats_chem: npt.NDArray[np.float_],
        alts_chem: npt.NDArray[np.float_],
        timesteps_chem: pd.DatetimeIndex,
        ):
    month = timesteps_chem[0].astype("datetime64[M]").astype(int) % 12 + 1
    bg_chem = xr.open_dataset("species.nc").sel(month=month-1)

    species = np.loadtxt('species_num.txt', dtype=str)

    # TEMP - FOR SPECIFYING PARTICULAR SPECIES TO INPUT
    given_numbers = [4,8,6,11,21,39,42,73,23,30,25,32,59,28,34,61,64,67,43,12,14,71,76,101,144,198,202]
    all_numbers = set(range(1, 219))  # Generate all numbers from 1 to 220
    
    missing_numbers = (all_numbers - set(given_numbers))
    missing_numbers = [x - 1 for x in missing_numbers]
    
    for i in missing_numbers:
            bg_chem[species[i]] = 0

    bg_chem = bg_chem * 1E+09 # convert mixing ratio to ppb

    # Downselect and interpolate bg_chem to high-res grid
    bg_chem = bg_chem.interp(longitude=lons_chem, latitude=lats_chem, level=units.m_to_pl(alts_chem), method="nearest")

    return bg_chem

##################### EMISSIONS #####################
# generate emissions dataset through trajectory generation, dry advection and plume-grid aggregation
def gen_emi(met: MetDataset,
            coords0: tuple[np.float_, np.float_, np.float_],
            flight_time: tuple[str, str],
            ts_fl: str,
            speed: float,
            theta: float,
            bbox: tuple[np.float_, np.float_, np.float_, np.float_, np.float_, np.float_],
            horiz_res_fl: float,
            vert_res_fl:float,
            sep_long: float,
            sep_lat: float,
            sep_alt: float,
            dt_integration: pd.Timedelta,
            max_age: pd.Timedelta,
            n_ac: int
            ) -> xr.Dataset:

    # init fl and dry_adv dicts
    fl_dict = {}
    dry_adv_dict = {}

    # Generate trajectories and do dry advection
    for i in range(0, n_ac):

        # generate flight paths
        fl_dict = traj_gen(i, fl_dict, met, coords0, flight_time, ts_fl, speed, theta, bbox, horiz_res_fl, vert_res_fl, sep_long, sep_lat, sep_alt)

        # do dry advection
        dry_adv_dict = dry_advection(i, dry_adv_dict, fl_dict, met, dt_integration, max_age)

        # convert fl_dict and dry_adv_dict to dataframes
        fl_dict[i] = fl_dict[i].dataframe
        dry_adv_dict[i] = dry_adv_dict[i].dataframe
        
        # calculate headings
        fl_dict[i] = calculate_heading_fl(fl_dict[i])
        dry_adv_dict[i] = calculate_heading_pl(dry_adv_dict[i])
               
        dry_adv_dict[i]["flight_id"] = fl_dict[i]["flight_id"][0]
        fl_dict[i]["waypoint"] = fl_dict[i].index

    # concatenate all dfs into one
    fl_df = pd.concat(fl_dict.values())
    dry_adv_df = pd.concat(dry_adv_dict.values())

    animate_fl(fl_df, dry_adv_df, bbox, dt_integration)

    plume_df = pd.merge(
                fl_df[['flight_id', 'waypoint', 'fuel_flow', 'true_airspeed', 'co2_m', 'h2o_m',
                'so2_m', 'nox_m', 'co_m', 'hc_m', 'nvpm_m']],
                dry_adv_df[['flight_id', 'waypoint', 'time', 'longitude', 'latitude', 'level', 'width', 'heading']], on=['flight_id', 'waypoint']).sort_values(by=['time', 'flight_id', 'waypoint'])
        
    plume_df['sin_a'] = np.sin(np.radians(plume_df['heading']))
    plume_df['cos_a'] = np.cos(np.radians(plume_df['heading']))
    
    plume_df['altitude'] = plume_df['level']
    
    plume_data = plume_to_grid(bbox, horiz_res_fl, plume_df, dt_integration)

    plume_data = plume_data.where(plume_data > 1E-30, 0.0)

    animate_pl(plume_data, "nox_m", 1)

    return plume_data

# trajectory generation
def traj_gen(i, fl_dict, met, coords, flight_time, ts_fl, speed, theta, bbox, horiz_res, vert_res, sep_long, sep_lat, sep_alt):

    if i == 0:
        # generate leader trajectory
        fl_dict[i] = leader_traj(coords, flight_time, ts_fl, speed, theta, bbox, horiz_res)
        fl_dict[i].attrs = {"flight_id": int(0), "aircraft_type": "A320"}
        fl_dict[i]["air_temperature"] = models.interpolate_met(met, fl_dict[i], "air_temperature")
        fl_dict[i]["specific_humidity"] = models.interpolate_met(met, fl_dict[i], "specific_humidity")

    else:
        # generate follower flights
        fl_dict[i] = follow_traj(coords, flight_time, ts_fl, speed, theta, bbox, horiz_res, sep_long*i, sep_lat*i, sep_alt*i)
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

    # get em mass per metre [g/m]
    fl_dict[i]["co2_m"] = (3.16 * fl_dict[i]["fuel_flow"] / fl_dict[i]["true_airspeed"]) / vert_res / 1E+03
    fl_dict[i]["h2o_m"] = (1.23 * fl_dict[i]["fuel_flow"] / fl_dict[i]["true_airspeed"]) / vert_res / 1E+03     
    fl_dict[i]["so2_m"] = (0.00084 * fl_dict[i]["fuel_flow"] / fl_dict[i]["true_airspeed"]) / vert_res / 1E+03
    fl_dict[i]["nox_m"] = (fl_dict[i]["nox_ei"] * fl_dict[i]["fuel_flow"] / fl_dict[i]["true_airspeed"]) / vert_res / 1E+03
    fl_dict[i]["co_m"] = (fl_dict[i]["co_ei"] * fl_dict[i]["fuel_flow"] / fl_dict[i]["true_airspeed"]) / vert_res / 1E+03
    fl_dict[i]["hc_m"] = (fl_dict[i]["hc_ei"] * fl_dict[i]["fuel_flow"] / fl_dict[i]["true_airspeed"]) / vert_res / 1E+03
    fl_dict[i]["nvpm_m"] = (fl_dict[i]["nvpm_ei_m"] * fl_dict[i]["fuel_flow"] / fl_dict[i]["true_airspeed"]) / vert_res / 1E+03

    return fl_dict

def leader_traj(coords, flight_time, ts_fl, speed, theta, bbox, horiz_res_fl):

    # Convert heading angle to radians
    theta_rad = np.radians(theta)

    # Calculate time array
    time = np.array(datalib.parse_timesteps(flight_time, freq=ts_fl), dtype="datetime64[s]")
    dt = (time - time[0]).astype('timedelta64[s]').astype(int)

    # Calculate trajectory points
    lats = coords[0] + ((speed * dt * np.cos(theta_rad))/6378E+03) * (180/np.pi)
    lons = coords[1] + ((speed * dt * np.sin(theta_rad))/6378E+03) * (180/np.pi) / np.cos(lats * np.pi/180)
    alts = coords[2] + np.zeros_like(time.astype(int))  # For simplicity, assume straight-line 
    # Filter points within bounding box
    mask = (
        (lats > bbox[0]-horiz_res_fl) & (lats < bbox[2]+horiz_res_fl) &
        (lons > bbox[1]-horiz_res_fl) & (lons < bbox[3]+horiz_res_fl) &
        (alts > bbox[4]) & (alts < bbox[5])
    )
   
    fl = Flight(longitude=lons[mask], latitude=lats[mask], altitude=alts[mask], time=time[mask])

    return fl

def follow_traj(coords, flight_time, ts_fl, speed, theta, bbox, horiz_res_fl, sep_long, sep_lat, sep_alt):

    theta_rad = np.radians(theta)
   
    # Calculate time array
    time = np.array(datalib.parse_timesteps(flight_time, freq=ts_fl), dtype="datetime64[s]")
    dt = (time - time[0]).astype('timedelta64[s]').astype(int)

    lat0 = coords[0] + ((sep_long*np.cos(theta_rad) - sep_lat*np.sin(theta_rad))/6378E+03) * (180/np.pi)
    lon0 = coords[1] + ((sep_long*np.sin(theta_rad) + sep_lat*np.cos(theta_rad))/6378E+03) * (180/np.pi) / np.cos((coords[0]) * np.pi/180)
    alt0 = coords[2] + sep_alt

    lats = lat0 + ((speed * dt * np.cos(theta_rad))/6378E+03) * (180/np.pi)
    lons = lon0 + ((speed * dt * np.sin(theta_rad))/6378E+03) * (180/np.pi) / np.cos(lats * np.pi/180)
    alts = alt0 + np.zeros_like(time.astype(int))  # For simplicity, assume straight-line 

    # Filter points within bounding box
    mask = (
        (lats > bbox[0]+horiz_res_fl) & (lats < bbox[2]-horiz_res_fl) &
        (lons > bbox[1]+horiz_res_fl) & (lons < bbox[3]-horiz_res_fl) &
        (alts > bbox[4]) & (alts < bbox[5])
    )

    fl = Flight(longitude=lons[mask], latitude=lats[mask], altitude=alts[mask], time=time[mask], flight_id=1)

    return fl

# dry advection
def dry_advection(i, dry_adv_dict, fl_dict, met, dt_integration, max_age):
            
    # do dry advection
    dt_integration = dt_integration
    max_age = max_age

    params = {
        "dt_integration": dt_integration,
        "max_age": max_age,
        "depth": 50.0,  # initial plume depth, [m]
        "width": 40.0,  # initial plume width, [m]
    }

    dry_adv = DryAdvection(met, params)

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

# plume to grid
def plume_to_grid(bbox, horiz_res_met, plume_df, dt_integration):

    # loop over time and plume property
    plume_data = {}
    
    for t, time in enumerate(plume_df['time'].unique()):
        
        # create geovectordataset to store instantaneous plume data
        plume_time_data = GeoVectorDataset(data=plume_df.loc[plume_df['time'] == time])
        calc_continuous(plume_time_data)
        
        plume_data[time] = {}
        
        # define molar masses of species g/mol
        mm = [46.01, 28.01, 44.01, 18.02, 64.07, 28.06] # g/mol
        NA = 6.022E+23 # Avogadro's number
        for i, p in enumerate(['nox_m', 'co_m']): #'co2_m', 'h2o_m', 'so2_m', 'hc_m']:
            
            # call contrails_to_hi_res_grid
            plume_property_data = contrails_to_hi_res_grid(time=time, 
                                                contrails_t=plume_time_data,
                                                var_name=p, 
                                                spatial_bbox=bbox, spatial_grid_res=horiz_res_met)
            
            plume_data[time][p] = (plume_property_data / 1E+03) * NA / mm[i] # convert to molecules/cm^3
        
            np.savetxt("plumes/plume_data_" + repr(t) + "_" + repr(p) + ".csv", plume_property_data, delimiter=",")    

        plume_data[time] = xr.Dataset(plume_data[time])
    
    # Convert plume_data dict to list so that it can be concatenated
    plume_data_list = [ds.assign_coords(time=key) for key, ds in plume_data.items()]
    plume_data = xr.concat(plume_data_list, dim="time")
    
    return plume_data

# animation
def animate_fl(fl_df, dry_adv_df, bbox, dt_integration):
    fig1, ax1 = plt.subplots()

    scat_fl = ax1.scatter([], [], s=5, c="red", label="Flight path")
    scat_pl = ax1.scatter([], [], s=0.1, c="blue", label="Plume evolution")
    ax1.legend(loc="upper left")
    ax1.set_xlim([bbox[0], bbox[2]])
    ax1.set_ylim([bbox[1], bbox[3]])
    
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
    ani.save(filename, dpi=300, writer=PillowWriter(fps=8))

def animate_pl(plume_data, species: str, i):
    fig, (ax, cbar_ax) = plt.subplots(1, 
                                      2, 
                                      gridspec_kw = {'width_ratios': (0.9, 0.05), 
                                                     'wspace': 0.2}, 
                                      figsize = (12, 8)
                                      )
    
    times = list(plume_data["time"])

    def heatmap_func(t):
        ax.cla()
        ax.set_title(t)

        plume_data[species].sel(time=t).transpose("latitude", "longitude").plot(
        ax = ax,
        cbar_ax = cbar_ax,
        add_colorbar = True,
        vmin = plume_data[species].min(),
        vmax = plume_data[species].max()
        )

    anim = FuncAnimation(fig = fig, func = heatmap_func, frames = times, blit = False)
    
    filename = pathlib.Path("plume" + repr(i) + ".gif")
    
    anim.save(filename, dpi=300, writer=PillowWriter(fps=8))

##################### TO CSVS #####################
# send dfs to csvs for chem analysis
def to_csvs(met, bg_chem, emi): 
    met_df = met.data.to_dask_dataframe(dim_order=['time', 'level', 'longitude', 'latitude'])
    
    #met_df = met.data.to_dataframe(dim_order=['time', 'level', 'longitude', 'latitude']).reset_index()

    met_df["latitude"] = met_df["latitude"].map("{:+08.3f}".format)
    met_df["longitude"] = met_df["longitude"].map("{:+08.3f}".format)
    met_df["sza"] = met_df["sza"].map("{:+0.3e}".format)
    # met_df = met_df.map_partitions(lambda df: df.apply(np.vectorize(lambda x: "{:0.3e}".format(x) if isinstance(x, (float, np.float32, np.float64)) else x)))
    met_df = met_df.apply((lambda x: x.map("{:0.3e}".format) if x.dtype in ['float32', 'float64'] else x), axis=1)

    # Convert bg_chem to df
    bg_chem_df = bg_chem.to_dask_dataframe(dim_order=['level', 'longitude', 'latitude'])

    #bg_chem_df = bg_chem.to_dataframe(dim_order=['level', 'longitude', 'latitude']).reset_index()
    
    bg_chem_df["latitude"] = bg_chem_df["latitude"].map("{:+08.3f}".format)
    bg_chem_df["longitude"] = bg_chem_df["longitude"].map("{:+08.3f}".format)
    bg_chem_df["month"] = bg_chem_df["month"].map("{:02d}".format)
    # bg_chem_df = bg_chem_df.map_partitions(lambda df: df.apply(np.vectorize(lambda x: "{:0.3e}".format(x) if isinstance(x, (float, np.float32, np.float64)) else x)))
    bg_chem_df = bg_chem_df.apply((lambda x: x.map("{:0.3e}".format) if x.dtype in ['float32', 'float64'] else x), axis=1) 
    
    # # Convert emi to df
    emi_df = emi.to_dask_dataframe(dim_order=['time', 'longitude', 'latitude']).fillna(0)

    #emi_df = emi.to_dataframe(dim_order=['time', 'longitude', 'latitude']).fillna(0).reset_index()
    
    emi_df["latitude"] = emi_df["latitude"].map("{:+08.3f}".format)
    emi_df["longitude"] = emi_df["longitude"].map("{:+08.3f}".format)
    # emi_df = emi_df.map_partitions(lambda df: df.apply(np.vectorize(lambda x: "{:0.3e}".format(x) if isinstance(x, (float, np.float32, np.float64)) else x)))
    emi_df = emi_df.apply((lambda x: x.map("{:0.3e}".format) if x.dtype in ['float32', 'float64'] else x), axis=1)

    # Remove temporary files if they exist
    if os.path.exists("met_df.csv"):
            os.remove("met_df.csv")
    if os.path.exists("bg_chem_df.csv"):
            os.remove("bg_chem_df.csv")
    if os.path.exists("emi_df.csv"):
            os.remove("emi_df.csv")

    # met_df_pd = met_df.compute().reset_index(drop=True)
    # bg_chem_df_pd = bg_chem_df.compute().reset_index(drop=True)
    # emi_df_pd = emi_df.compute().reset_index(drop=True)

    # Write DataFrame 1 to the temporary file
    met_df.to_csv("met_df.csv", single_file=True, index=False)

    # Write DataFrame 2 to the temporary file
    bg_chem_df.to_csv("bg_chem_df.csv", single_file=True, index=False)

    # Write DataFrame 3 to the temporary file
    emi_df.to_csv("emi_df.csv", single_file=True, index=False)

    return

#################### RUN BOXM #####################
# run the photochemical box model
def run_boxm(lons_chem, lats_chem, timesteps_chem, ts_chem):
        
        ncell = len(lats_chem) * len(lons_chem) # * len(alts_chem)
        nts = len(timesteps_chem) - 1
        dts = int(timeparse(ts_chem))
        print(ncell, nts, dts)

        boxm_f2py.init(ncell)

        for t in range(nts):
                # if t*dts % 3600 == 0:
                print("Time: ", t)

                boxm_f2py.read(ncell)
                boxm_f2py.calc_aerosol()
                boxm_f2py.chemco()
                boxm_f2py.calc_j(ncell)
                boxm_f2py.photol()

                if t != 0:
                        boxm_f2py.deriv(dts)

                boxm_f2py.write(dts, ncell)

        boxm_f2py.deallocate()


if __name__=="__main__":
    main()