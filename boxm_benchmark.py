import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime
from pycontrails.datalib.ecmwf import ERA5
from init_chem import CHEM
from chem import ChemDataset
from boxm_f2py import boxm_f2py
#from boxmpy import boxmpy
from pycontrails.core import datalib
from pycontrails.physics import geo, thermo, units, constants
import numpy as np
import itertools
import os
from pytimeparse.timeparse import timeparse
import tempfile
import numpy as np


# This file needs to generate a common input to feed to both original BOXM and new f2py implementation. The aim is to generate two sets  of outputs (species concs over time), and to diff these outputs to see how different they are to each other.

# Hard coded inputs
time = ("2022-01-20 12:00:00", "2022-01-21 12:00:00")
lon_bounds = (-32.5, -31.5) 
lat_bounds = (47.5, 48.5)
alt_bounds = (12000, 12500)
horiz_res = 1
vert_res = 500
ts_met = "6H"
ts_disp = "1min"
ts_chem = "20s"

species = np.loadtxt('species_num.txt', dtype=str)

def main():
        
        # Create met, bg_chem, emi datasets and send to files
        lats, lons, alts, timesteps, met, bg_chem, emi = preprocess()

        run_boxm(met, bg_chem, timesteps, lats, lons, alts)

        ncell = len(lats) * len(lons) * len(alts)
        nts = len(timesteps) - 1
        dts = int(timeparse(ts_chem))
        print(ncell, nts, dts)

        boxm_f2py.init(ncell)

        for t in range(nts):
                if t*dts % 3600 == 0:
                        print("Time: ", t)

                boxm_f2py.read(ncell)
                boxm_f2py.calc_aerosol()
                boxm_f2py.chemco()
                boxm_f2py.calc_j(ncell)
                boxm_f2py.photol()

                if t != 0:
                        boxm_f2py.deriv(dts)

                boxm_f2py.write(dts)

        boxm_f2py.deallocate()

def preprocess():
        met_pressure_levels = np.array([400, 300, 200, 100])

        lons = np.arange(lon_bounds[0], lon_bounds[1], horiz_res)
        lats = np.arange(lat_bounds[0], lat_bounds[1], horiz_res)
        alts = np.arange(alt_bounds[0], alt_bounds[1], vert_res)
        timesteps = np.array(datalib.parse_timesteps(time, freq=ts_chem), dtype="datetime64[ns]")
        

        # Import met data from ERA5
        era5 = ERA5(
                # time=time,
                time=time[0],
                timestep_freq=ts_met,
                variables=[
                        "t",
                        "q",
                        "relative_humidity"
                ],
                grid=1.0,
                url="https://cds.climate.copernicus.eu/api/v2",
                key="171715:93148f05-a469-43a8-ae25-44c8eba81e90",
                pressure_levels=met_pressure_levels
        )

        # download data from ERA5 (or open from cache)
        met = era5.open_metdataset()
        met.data = met.data.transpose("latitude", "longitude", "level", "time", ...)
        print(met.data["air_temperature"])
        month = timesteps[0].astype("datetime64[M]").astype(int) % 12 + 1
        bg_chem = xr.open_dataset("species.nc").sel(month=month-1) 

        species = np.loadtxt('species_num.txt', dtype=str)

        # TEMP - FOR SPECIFYING PARTICULAR SPECIES TO INPUT
        given_numbers = [4,8,6,11,21,39,42,73,23,30,25,32,59,28,34,61,64,67,43,12,14,71,76,101,144,198,202]
        all_numbers = set(range(1, 219))  # Generate all numbers from 1 to 220
        
        missing_numbers = (all_numbers - set(given_numbers))
        missing_numbers = [x - 1 for x in missing_numbers]
      
        for i in missing_numbers:
                bg_chem[species[i]] = 0

        bg_chem = bg_chem * 1E+09
        
        emi = emissions(lats, lons, alts, timesteps)

        # Downselect and interpolate MetDataset to high-res grid
        met.data = met.data.interp(longitude=lons, latitude=lats, level=units.m_to_pl(alts), time=timesteps, method="nearest")

        met = calc_M_H2O(met)

        met.data["sza"] = (('latitude', 'longitude', 'time'), calc_sza(lats, lons, timesteps))
        
        # fix for comparison
        met.data["air_temperature"] = met.data["air_temperature"][:,:,:,0]
        met.data["specific_humidity"] = met.data["specific_humidity"][:,:,:,0]
        met.data["relative_humidity"] = met.data["relative_humidity"][:,:,0]
        met.data["M"] = met.data["M"][:,:,:,0]
        met.data["H2O"] = met.data["H2O"][:,:,:,0]
        met.data["O2"] = met.data["O2"][:,:,:,0]
        met.data["N2"] = met.data["N2"][:,:,:,0]

        # Downselect and interpolate bg_chem to high-res grid
        bg_chem = bg_chem.interp(longitude=lons, latitude=lats, level=units.m_to_pl(alts), method="nearest")

        # # Downselect and interpolate emi ChemDataset to high-res grid
        # emi.data = emi.data.interp(longitude=lons, latitude=lats, level=units.m_to_pl(alts), time=timesteps, method="nearest")

        pd.set_option('display.max_rows', 500)

        # Convert met to df
        met_df = met.data.to_dataframe(dim_order=['time', 'level', 'longitude', 'latitude']).reset_index()
        
        met_df["latitude"] = met_df["latitude"].map("{:+08.3f}".format)
        met_df["longitude"] = met_df["longitude"].map("{:+08.3f}".format)
        met_df["sza"] = met_df["sza"].map("{:+0.3e}".format)
        met_df = met_df.apply(lambda x: x.map("{:0.3e}".format) if x.dtype in ['float32', 'float64'] else x)

        # Convert bg_chem to df
        bg_chem_df = bg_chem.to_dataframe(dim_order=['level', 'longitude', 'latitude']).reset_index()
        
        bg_chem_df["latitude"] = bg_chem_df["latitude"].map("{:+08.3f}".format)
        bg_chem_df["longitude"] = bg_chem_df["longitude"].map("{:+08.3f}".format)
        bg_chem_df["month"] = bg_chem_df["month"].map("{:02d}".format)
        bg_chem_df = bg_chem_df.apply(lambda x: x.map("{:0.3e}".format) if x.dtype in ['float32', 'float64'] else x)
        
        # # Convert emi to df
        emi_df = emi.data.to_dataframe(dim_order=['time', 'level', 'longitude', 'latitude']).fillna(0).reset_index().drop(columns=["air_pressure", "altitude"])
        
        emi_df["latitude"] = emi_df["latitude"].map("{:+08.3f}".format)
        emi_df["longitude"] = emi_df["longitude"].map("{:+08.3f}".format)
        emi_df = emi_df.apply(lambda x: x.map("{:0.3e}".format) if x.dtype in ['float32', 'float64'] else x)

        # Remove temporary files if they exist
        if os.path.exists("met_df.csv"):
                os.remove("met_df.csv")
        if os.path.exists("bg_chem_df.csv"):
                os.remove("bg_chem_df.csv")
        if os.path.exists("emi_df.csv"):
                os.remove("emi_df.csv")

        # Write DataFrame 1 to the temporary file
        met_df.to_csv("met_df.csv", index=False)

        # Write DataFrame 2 to the temporary file
        bg_chem_df.to_csv("bg_chem_df.csv", index=False)

        # Write DataFrame 3 to the temporary file
        emi_df.to_csv("emi_df.csv", index=False)

        return lats, lons, alts, timesteps, met, bg_chem, emi

def emissions(lats, lons, alts, timesteps):
        
        dataarrays = {}
        for s in ['NO2', 'CO', 'HCHO', 'CH3CHO', 'CH3COCH3', 'C2H6', 'C2H4', 'C3H8', 'C3H6', 'C2H2', 'BENZENE', 'TOLUENE', 'C2H5CHO']:
                dataarrays[s] = xr.DataArray(
                np.zeros((len(lats), len(lons), len(alts), len(timesteps))),
                dims=["latitude", "longitude", "level", "time"],
                coords={
                "latitude": lats,
                "longitude": lons,
                "level": alts,
                "time": timesteps,
                },
                name=s,
        )
                
        emi = xr.Dataset(dataarrays)
        emi = ChemDataset(emi)
        emi.data = emi.data.transpose("latitude", "longitude", "level", "time", ...)

        return emi

def calc_M_H2O(met):

        """Calculate number density of air molecules at each pressure level M"""
        N_A = 6.022e23 # Avogadro's number
        
        # Get air density from pycontrails physics.thermo script
        rho_d = met["air_pressure"].data / (constants.R_d * met["air_temperature"].data)

        # Calculate number density of air (M) to feed into box model calcs
        met.data["M"] = (N_A / constants.M_d) * rho_d * 1e-6  # [molecules / cm^3]
        met.data["M"] = met.data["M"].transpose("latitude", "longitude", "level", "time")
                
        # Use expand_dims to add the new "species" dimension
        #self.data["M"] = self.data["M"].expand_dims(dim={'species': self.data["species"]}, axis=4)


        # Calculate H2O number concentration to feed into box model calcs
        met.data["H2O"] = (met["specific_humidity"].data / constants.M_v) * N_A * rho_d * 1e-6 
        # [molecules / cm^3]

        # Calculate O2 and N2 number concs based on M
        met.data["O2"] = 2.079E-01 * met.data["M"]
        met.data["N2"] = 7.809E-01 * met.data["M"] 

        return met

def calc_sza(lats, lons, timesteps):
        """Calculate szas for each cell at all timesteps"""
        sza = np.zeros((len(lats), len(lons), len(timesteps)))

        for lon, lonval in enumerate(lons):
                for lat, latval in enumerate(lats):
                        
                        theta_rad = geo.orbital_position(timesteps)
                        
                        sza[lat, lon, :] = np.arccos(geo.cosine_solar_zenith_angle(lonval, latval, timesteps, theta_rad))
        return sza

def run_boxm(met, bg_chem, timesteps, lats, lons, alts):
        start_time = timesteps[0]
        print(start_time)
        end_time = timesteps[-1]
        
        SPECIES = ["NO2", "NO", "O3", "CO", "CH4", "HCHO", "CH3CHO", "CH3COCH3",
                           "C2H6", "C2H4", "C3H8", "C3H6", "C2H2", "NC4H10", "TBUT2ENE",
                           "BENZENE", "TOLUENE", "OXYL", "C5H8", "H2O2", "HNO3", "C2H5CHO",
                           "CH3OH", "MEK", "CH3OOH", "PAN", "MPAN"]
        
        for lat, lon, alt in itertools.product(lats[0:1], lons[0:1], alts[0:1]):
                start_time = np.datetime64(start_time)  # Convert start_time to NumPy datetime object

                DAY = 1 # start_time.astype("datetime64[D]").astype(int) % 365 + 1
                MONTH = start_time.astype("datetime64[M]").astype(int) % 12 + 1
                YEAR = start_time.astype("datetime64[Y]").astype(int) + 1970
                LEVEL = get_pressure_level(alt, met)
                longbox = longitude_to_longbox(lon) + 1
                latbox = latitude_to_latbox(lat) + 1
                RUNTIME = 31 # (end_time - start_time).astype("datetime64[D]").astype(int) % 365 + 1

                M = met["M"].data.sel(longitude=lon, latitude=lat, level=units.m_to_pl(alt)).values.item() #, time=start_time).values.item()
                P = units.m_to_pl(alt).item()
                H2O = met["H2O"].data.sel(longitude=lon, latitude=lat, level=units.m_to_pl(alt)).values.item() #, time=start_time).values.item()
                TEMP = met["air_temperature"].data.sel(longitude=lon, latitude=lat, level=units.m_to_pl(alt)).values.item() #, time=start_time).values.item()

                print(M, P, H2O, TEMP)

                BOXMinput = open("newBOXMODEL.IN", "w")
                # BOXMinput = open("BOXMinput_" + repr(lon) + "_" + repr(lat) + "_" + repr(alt) + ".in", "w")

                BOXMinput.write(repr(DAY) + "\n" + repr(MONTH) + "\n" + repr(YEAR) + "\n" + repr(LEVEL) + "\n" + repr(longbox) + "\n" + repr(latbox) + "\n" + repr(RUNTIME) + "\n" + repr(M) + "\n" + repr(P) + "\n" + repr(H2O) + "\n" + repr(TEMP) + "\n")

                for s in SPECIES:
                        BOXMinput.write(repr(bg_chem[s].loc[lat, lon, units.m_to_pl(alt)].values.item()) + "\n")
                        #      .sel(species=s, longitude=lon, latitude=lat, time=start_time).sel(level=units.m_to_pl(alt), method="nearest").values.item()) + "\n")
                        
                        print(repr(s) + " " + repr(bg_chem[s].loc[lat, lon, units.m_to_pl(alt)].values.item()))

                        # print(repr(s) + " " + repr(chem["Y"].data.sel(species=s, longitude=lon, latitude=lat, time=start_time).sel(level=units.m_to_pl(alt), method="nearest").values.item()))
                
                BOXMinput.close()

def get_pressure_level(alt, met):
        # Convert alt to pressure level (hPa)``
        chem_pressure_levels = np.array([962, 861, 759, 658, 556, 454, 353, 251, 150.5])

        # Convert altitude to pressure using a standard atmosphere model
        pressure = units.m_to_pl(alt)

        # Find the index of the closest value in the array
        idx = (np.abs(chem_pressure_levels - pressure)).argmin()

        return idx

def latitude_to_latbox(latitude):
        # Map the latitude to the range 0-1
        normalized_latitude = (latitude + 87.5) / 180

        # Map the normalized latitude to the range 1-72
        latbox = normalized_latitude * 36

        # Round to the nearest integer and return
        return round(latbox)

def longitude_to_longbox(longitude):
        # Map the longitude to the range 0-1
        normalized_longitude = (longitude + 177.5) / 360

        # Map the normalized longitude to the range 1-144
        longbox = normalized_longitude * 72

        # Round to the nearest integer and return
        return round(longbox)

if __name__=="__main__":
        main()
