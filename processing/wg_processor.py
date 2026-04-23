import argparse
import glob
import json
import os
import re
import sys
from typing import Optional, Dict

import numpy as np
import pandas as pd
import xarray as xr

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================

KNOTS_TO_MS = 0.514444
KNOTS_TO_MPS_ADCP = 1852 / 3600

DATA_CONFIG = {
    'waves': {
        'script_name': 'waveglider-waves', 
        'json_key': None, 
        'timestamp_col': 'timeStamp', 
        'to_float': True
    },
    'weather': {
        'script_name': 'waveglider-weather', 
        'json_key': None, 
        'timestamp_col': 'timeStamp', 
        'to_float': True
    },
    'adcp': {
        'script_name': 'waveglider-adcp', 
        'json_key': None, 
        'timestamp_col': 'timeStamp', 
        'to_float': False
    },
    'telemetry': {
        'script_name': 'waveglider-telemetry', 
        'json_key': None, 
        'timestamp_col': 'timeStamp', 
        'to_float': False
    }
}

# ==========================================
# DATA LOADERS
# ==========================================

def load_waveglider_data(data_type: str, base_dir: str) -> Optional[pd.DataFrame]:
    """
    Loads raw JSON records from a directory and converts them to a Pandas DataFrame.

    Args:
        data_type: Key in DATA_CONFIG (e.g., 'waves', 'weather').
        base_dir: Directory containing the JSON files.

    Returns:
        A cleaned DataFrame indexed by time, or None if no files are found.
    """

    config = DATA_CONFIG[data_type]
    search_path = os.path.join(base_dir, f"{config['script_name']}_*.json")
    file_list = sorted(glob.glob(search_path))
    
    if not file_list:
        return None

    all_data_records = []
    for filename in file_list:
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                records = data[config['json_key']] if config['json_key'] else data
                if records:
                    all_data_records.extend(records)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Error parsing {filename}: {e}")

    if not all_data_records:
        return None

    df = pd.DataFrame(all_data_records)
    
    # Standardize Time Index
    df['time'] = pd.to_datetime(df[config['timestamp_col']])
    df.set_index('time', inplace=True)
    df = df[~df.index.duplicated(keep='first')] 
    df = df.drop(columns=[config['timestamp_col']]) 
    
    if config['to_float']:
        df = df.apply(pd.to_numeric, errors='coerce')
        
    return df

# ==========================================
# FORMATTING FUNCTIONS
# ==========================================

def format_weather(df: pd.DataFrame, sensor_cfg: Dict) -> xr.Dataset:
    """Processes Airmar weather data and sets the primary GPS track."""
    ds = xr.Dataset.from_dataframe(df).drop_duplicates(dim='time')

    # Unit Conversions
    ds['WIND_SPEED_AIRMAR_MEAN'] = ds['avgWindSpeed(kt)'] * KNOTS_TO_MS
    ds['WIND_SPEED_AIRMAR_STDDEV'] = ds['stdDevWindSpeed(kt)'] * KNOTS_TO_MS
    ds['GUST_WND_AIRMAR_MEAN'] = ds['gustSpeed(kt)'] * KNOTS_TO_MS

    rename_map = {
        'avgTemp(C)': 'TEMP_AIR_AIRMAR_MEAN',
        'avgPress(mbar)': 'BARO_PRES_AIRMAR_MEAN',
        'avgWindDir(deg)': 'WIND_FROM_AIRMAR_MEAN',
        'stdDevWindDir(deg)': 'WIND_FROM_AIRMAR_STDDEV',
        'gustDir(deg)': 'GUST_WND_DIR_AIRMAR_MEAN'
    }
    ds = ds.rename(rename_map)

    # Cleanup internal variables
    drop_cols = [
        'avgWindSpeed(kt)', 'stdDevWindSpeed(kt)', 'gustSpeed(kt)', 
        'gustTime(hour)', 'gustTime(minute)', 'numValidWindSamples'
    ]
    ds = ds.drop_vars(drop_cols, errors='ignore')

    # Apply Metadata
    airmar_meta = sensor_cfg.get('AIRMAR', {})
    for var in ds.data_vars:
        ds[var].attrs.update(airmar_meta)
        
    return ds.set_coords(['latitude', 'longitude'])

def format_waves(df: pd.DataFrame, sensor_cfg: Dict) -> xr.Dataset:
    """Processes WaveGlider wave sensor data."""
    ds = xr.Dataset.from_dataframe(df).drop_duplicates(dim='time')
    
    ds = ds.rename({
        'hs (m)': 'WAVE_SIGNIFICANT_HEIGHT',
        'ta (s)': 'WAVE_AVERAGE_PERIOD',
        'tp (s)': 'WAVE_DOMINANT_PERIOD',
        'dp (deg)': 'WAVE_DOMINANT_DIRECTION'
    })
    
    # Data Validation
    ds["WAVE_DOMINANT_DIRECTION"] = ds["WAVE_DOMINANT_DIRECTION"].where(ds["WAVE_DOMINANT_DIRECTION"] <= 360)
    
    wave_meta = sensor_cfg.get('WAVES', {})
    for var in ds.data_vars:
        ds[var].attrs.update(wave_meta)
        
    drop_cols = ['fs (Hz)', 'sample Gaps', 'samples']
    return ds.drop_vars(drop_cols, errors='ignore').set_coords(['latitude', 'longitude'])

def format_adcp(df: pd.DataFrame, sensor_cfg: Dict) -> xr.Dataset:
    """
    Stacks ADCP bins into a 2D (time, depth) Xarray Dataset and calculates 
    East/North current components.
    """
    ds_raw = df.to_xarray().rename({"bin1 distance (m)": "bin 1 Depth"}).drop_duplicates(dim='time')
    
    # Identify bin columns using regex
    def get_bins(keyword):
        return sorted(
            [v for v in ds_raw.data_vars if keyword in v], 
            key=lambda x: int(re.search(r'bin (\d+)', x).group(1))
        )

    mag_vars = get_bins('Mag')
    dir_vars = get_bins('Dir')
    depth_vars = get_bins('Depth')

    def stack_vars(var_list):
        arrays = [xr.DataArray(pd.to_numeric(ds_raw[var].values, errors='coerce'), 
                               dims=['time'], coords={'time': ds_raw.time}) for var in var_list]
        return xr.concat(arrays, dim='depth_bin')

    magnitude = stack_vars(mag_vars) * KNOTS_TO_MPS_ADCP
    direction_rad = np.deg2rad(stack_vars(dir_vars))
    depth = stack_vars(depth_vars)
    
    ds_2d = xr.Dataset({
        'WATER_CURRENT_EAST_MEAN': magnitude * np.sin(direction_rad),
        'WATER_CURRENT_NORTH_MEAN': magnitude * np.cos(direction_rad)
    })
    
    # Set depth as a coordinate
    mean_depths = depth.mean(dim='time', skipna=True)
    ds_2d = ds_2d.assign_coords(depth=('depth_bin', mean_depths.values.round()))
    ds_2d = ds_2d.swap_dims({'depth_bin': 'depth'})

    # Attach GPS from raw df if available
    for c in ['latitude', 'longitude']:
        if c in ds_raw:
            ds_2d = ds_2d.assign_coords({c: ds_raw[c].astype(float)})

    adcp_meta = sensor_cfg.get('ADCP', {})
    for var in ds_2d.data_vars:
        ds_2d[var].attrs.update(adcp_meta)
        
    return ds_2d

def format_airsea_csv(csv_path: str, sensor_cfg: Dict) -> xr.Dataset:
    """Parses the main AirSea CSV, renames variables, and applies CF-compliant metadata."""
    print("Formatting AirSea CSV...")
    df = pd.read_csv(csv_path)

    df['Logger_DateTime'] = pd.to_datetime(df['Logger_DateTime'], format='%Y-%m-%d %H:%M:%S%f', errors='coerce')
    df = df.dropna(subset=['Logger_DateTime']).sort_values('Logger_DateTime')
    df = df.drop_duplicates(subset=['Logger_DateTime'], keep='first')
    
    ds = df.set_index('Logger_DateTime').to_xarray()
    ds = ds.drop_vars(['Transmission_Time', 'index'], errors='ignore').rename({'Logger_DateTime': 'time'})
    
    # Resample to 10-minute bins to reduce noise/align with other sensors
    ds = ds.resample(time="10min").mean()

    """ The following variables are also available in the CSV:
    Transmission_Time
    RBR_SeaPressure_Avg	
    RBR_Depth_Avg	
    RBR_Temp_Correction	
    rain_amount_Avg	
    rain_duration_Avg	
    rain_peak_Avg
    """
    
    rename_dict = {
        'atmospheric_temperature': 'TEMP_AIR_WXT_MEAN',
        'atmospheric_pressure_Avg': 'BARO_PRES_WXT_MEAN',
        'relative_humidity_Avg': 'RH_WXT_MEAN',
        'wind_speed_Avg': 'WIND_SPEED_WXT_MEAN',
        'wind_direction_Avg': 'WIND_FROM_WXT_MEAN',
        'RBR_Temp_Avg': 'TEMP_WATER_LEGATO_MEAN',
        'RBR_Conductivity_Avg': 'COND_LEGATO_MEAN',
        'RBR_Pressure_Avg': 'PRES_LEGATO_MEAN',
        'RBR_Salinity_Avg': 'SAL_LEGATO_MEAN',
        'CODA_Oxygen_Avg': 'O2_CONC_CODA_MEAN',
        'CODA_Temp_Avg': 'TEMP_O2_CODA_MEAN',
        'CODA_Phase_Avg': 'O2_PHASE_CODA_MEAN',      
        'Cyclops_Chlorophyll_Avg': 'CHLOR_CYCLOPS_MEAN',
        'Ux_gill_Avg': 'WIND_U_GILL_MEAN',
        'Uy_gill_Avg': 'WIND_V_GILL_MEAN',
        'Uz_gill_Avg': 'WIND_W_GILL_MEAN',
        'rain_intensity_Avg': 'RAIN_INTENSITY_WXT_MEAN',
        'RBR_MeasurementCount': 'RBR_MEASUREMENT_COUNT'
    }
    ds = ds.rename({k: v for k, v in rename_dict.items() if k in ds})
    
    # Drop unused diagnostic vars
    drop_vars = [
        'RBR_Temp_Correction', 'RBR_SeaPressure_Avg', 
        'RBR_Depth_Avg', 'rain_amount_Avg', 'rain_duration_Avg', 'rain_peak_Avg'
    ]
    ds = ds.drop_vars(drop_vars, errors='ignore')

    # Apply CF Standard Names and Units
    meta_map = {
        'TEMP_AIR_WXT_MEAN': {"standard_name": "air_temperature", "units": "degree_C", "cfg": "WXT536"},
        'BARO_PRES_WXT_MEAN': {"standard_name": "air_pressure", "units": "hPa", "cfg": "WXT536"},
        'RH_WXT_MEAN': {"standard_name": "relative_humidity", "units": "percent", "cfg": "WXT536"},
        'WIND_SPEED_WXT_MEAN': {"standard_name": "wind_speed", "units": "m s-1", "cfg": "WXT536"},
        'WIND_FROM_WXT_MEAN': {"standard_name": "wind_from_direction", "units": "degree", "cfg": "WXT536"},
        'WIND_U_GILL_MEAN': {"standard_name": "eastward_wind", "units": "m s-1", "cfg": "GILL"},
        'WIND_V_GILL_MEAN': {"standard_name": "northward_wind", "units": "m s-1", "cfg": "GILL"},
        'WIND_W_GILL_MEAN': {"standard_name": "upward_air_velocity", "units": "m s-1", "cfg": "GILL"},
        'TEMP_WATER_LEGATO_MEAN': {"standard_name": "sea_water_temperature", "units": "degree_C", "cfg": "LEGATO"},
        'SAL_LEGATO_MEAN': {"standard_name": "sea_water_practical_salinity", "units": "1", "cfg": "LEGATO"},
        'O2_CONC_CODA_MEAN': {"standard_name": "mole_concentration_of_dissolved_molecular_oxygen_in_sea_water", "units": "mmol m-3", "cfg": "CODA"},
        'CHLOR_CYCLOPS_MEAN': {"standard_name": "mass_concentration_of_chlorophyll_in_sea_water", "units": "ug L-1", "cfg": "CYCLOPS"},
        'RAIN_INTENSITY_WXT_MEAN': {"standard_name": "rain_intensity", "units": "mm hr-1", "cfg": "WXT536"}
    }

    for var, attrs in meta_map.items():
        if var in ds:
            sensor_key = attrs.pop("cfg")
            ds[var].attrs = {**attrs, **sensor_cfg.get(sensor_key, {})}

    return ds.drop_duplicates(dim='time')

# ==========================================
# MASTER BUILDER
# ==========================================

def build_master_dataset(input_dir: str, config_dict: Dict) -> Optional[xr.Dataset]:
    """
    Main orchestration logic. Merges disparate sensors onto a common time axis
    and interpolates GPS coordinates.
    """
    sensor_cfg = config_dict.get('sensors', {})
    mission_cfg = config_dict.get('mission_config', {})
    mission_name = mission_cfg.get('mission_name', 'Unknown Mission')
    
    print(f"--- Building Dataset for {mission_name} ---")

    # 1. Load Weather (Source of Truth for GPS)
    df_weather = load_waveglider_data('weather', input_dir)
    if df_weather is None:
        print("❌ CRITICAL: Weather data missing. Cannot establish GPS track.")
        return None
    ds_master = format_weather(df_weather, sensor_cfg)
    
    # 2. Collect other datasets
    datasets_to_merge = []
    
    # AirSea CSV
    csv_path = os.path.join(input_dir, "Decoded_AirSea_Data.csv")
    if os.path.exists(csv_path):
        try:
            ds_csv = format_airsea_csv(csv_path, sensor_cfg)
            datasets_to_merge.append(ds_csv.drop_vars(['latitude', 'longitude'], errors='ignore'))
        except Exception as e:
            print(f"⚠️ AirSea CSV Error: {e}")

    # Waves & ADCP
    loaders = [('waves', format_waves), ('adcp', format_adcp)]
    for dtype, formatter in loaders:
        df = load_waveglider_data(dtype, input_dir)
        if df is not None:
            ds = formatter(df, sensor_cfg)
            datasets_to_merge.append(ds.drop_vars(['latitude', 'longitude'], errors='ignore'))

    # 3. Merge and Interpolate
    print("Merging datasets and interpolating GPS...")
    for ds in datasets_to_merge:
        ds_master = ds_master.combine_first(ds)

    # Fill GPS gaps created by merging high-frequency sensors with lower-frequency GPS
    ds_master['latitude'] = ds_master['latitude'].interpolate_na(dim='time', method='linear', limit=10)
    ds_master['longitude'] = ds_master['longitude'].interpolate_na(dim='time', method='linear', limit=10)

    # 4. Slicing & Cleanup
    start_time = mission_cfg.get('start_time')
    end_time = mission_cfg.get('end_time')
    if start_time or end_time:
        ds_master = ds_master.sel(time=slice(start_time, end_time))

    # Drop variables that are all NaN (offline sensors)
    empty_vars = [v for v in ds_master.data_vars if ds_master[v].isnull().all()]
    if empty_vars:
        print(f"Dropping empty sensors: {empty_vars}")
        ds_master = ds_master.drop_vars(empty_vars)

    # 5. Global Metadata
    ds_master['time'].attrs = {"standard_name": "time", "axis": "T"}
    
    now_iso = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
    dynamic_attrs = {
        "title": f"Waveglider {mission_name} Mission",
        "project": mission_name,
        "id": f"WG_{mission_name.replace(' ', '_')}_L1",
        "history": f"Created {now_iso} by merging JSON/CSV sources.",
        "time_coverage_start": str(ds_master.time.min().values),
        "time_coverage_end": str(ds_master.time.max().values),
        "geospatial_lat_min": float(ds_master.latitude.min()),
        "geospatial_lat_max": float(ds_master.latitude.max()),
        "geospatial_lon_min": float(ds_master.longitude.min()),
        "geospatial_lon_max": float(ds_master.longitude.max()),
    }
    # --- DYNAMIC BOUNDS & RICH METADATA ---
    ds_master['time'].attrs = {"standard_name": "time", "axis": "T"}
    
    # Define coordinate metadata as requested
    lat_attrs = {
        "long_name": "latitude of each measurement and GPS location",
        "standard_name": "latitude",
        "units": "degrees_north",
        "valid_min": -90.0,
        "valid_max": 90.0,
        "axis": "Y",
        "comment": "Source: Airmar 200WX"
    }
    
    lon_attrs = {
        "long_name": "longitude of each measurement and GPS location",
        "standard_name": "longitude",
        "units": "degrees_east",
        "valid_min": -180.0,
        "valid_max": 180.0,
        "axis": "X",
        "comment": "Source: Airmar 200WX"
    }

    if 'latitude' in ds_master:
        # Fill remaining NaNs with the requested FillValue and set attributes
        ds_master['latitude'] = ds_master['latitude'].fillna(-9999.9)
        ds_master['latitude'].attrs = lat_attrs
        # Standard Xarray encoding for the NetCDF _FillValue
        ds_master['latitude'].encoding["_FillValue"] = -9999.9

    if 'longitude' in ds_master:
        ds_master['longitude'] = ds_master['longitude'].fillna(-9999.9)
        ds_master['longitude'].attrs = lon_attrs
        ds_master['longitude'].encoding["_FillValue"] = -9999.9

    ds_master.attrs = {**config_dict.get('global_attributes', {}), **dynamic_attrs}

    return ds_master

# ==========================================
# CLI ENTRY POINT
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wave Glider CF-NetCDF Processor")
    parser.add_argument('-i', '--input', required=True, help="Input directory")
    parser.add_argument('-o', '--output', required=True, help="Output NetCDF path")
    parser.add_argument('-c', '--config', required=True, help="Path to config.json")
    
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            metadata_cfg = json.load(f)
    except Exception as e:
        print(f"❌ Config Error: {e}")
        sys.exit(1)

    final_ds = build_master_dataset(args.input, metadata_cfg)
    
    if final_ds is not None:
        final_ds.to_netcdf(args.output, format="NETCDF4")
        print(f"✅ Saved: {args.output}")
    else:
        sys.exit(1)
