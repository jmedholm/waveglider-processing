import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr
import string
import json
import pandas as pd
import gsw_xarray as gsw
from cmocean import cm as cmo  
import sys
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob
import re

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Define color variables with hexadecimal color codes
wgc = '#9B765E' # Wave Glider's colour
plt.rcParams['font.size'] = 10

ds = xr.open_dataset('/home/jedholm/share/gliders/waveglider/wg1170/WG_WHIRLS_M2_L1.nc').interpolate_na(dim='time', method='nearest')

# 2. Apply your initial cleaning filters
ds = ds.where(ds > -500)
#ds = ds.where(ds["RBR_MEASUREMENT_COUNT"] == 1)

# 3. Apply your rolling median outlier filter
for var in ds.data_vars:
    mean, std = ds[var].mean(), ds[var].std()
    ds[var] = ds[var].where(abs(ds[var] - mean) < 4 * std).rolling(time=5, center=True).median()

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from datetime import datetime

plot_gill = True

# --- Create Figure ---
fig, axs = plt.subplots(
    nrows=4 if plot_gill else 3, ncols=1,
    figsize=(12, 10),
    sharex=True,
    constrained_layout=True
)

lw_thin = 1
lw_thick = 2
alpha_raw = 0.5

# --- 1. Data Processing (Smoothed for Dashboard) ---
# Pre-calculating to keep the plotting calls clean
airmar_wind_smooth = ds["WIND_SPEED_AIRMAR_MEAN"].resample(time="10min").mean().rolling(time=6, center=True).mean()
vaisala_wind_smooth = ds["WIND_SPEED_WXT_MEAN"].resample(time="10min").mean().rolling(time=6, center=True).mean()
airmar_pres_smooth = ds["BARO_PRES_AIRMAR_MEAN"].resample(time="10min").mean().rolling(time=6, center=True).mean()
vaisala_pres_smooth = ds["BARO_PRES_WXT_MEAN"].resample(time="10min").mean().rolling(time=6, center=True).mean()
wave_smooth = ds["WAVE_SIGNIFICANT_HEIGHT"].resample(time="20min").mean().rolling(time=6, center=True).mean()

# Pre-calculating to keep the plotting calls clean
airmar_air_temp_smooth = ds["TEMP_AIR_AIRMAR_MEAN"].resample(time="10min").mean().rolling(time=6, center=True).mean()
vaisala_air_temp_smooth = ds["TEMP_AIR_WXT_MEAN"].resample(time="10min").mean().rolling(time=6, center=True).mean()
rbr_sst_smooth = ds["TEMP_WATER_LEGATO_MEAN"].where(ds["RBR_MEASUREMENT_COUNT"] == 1).resample(time="10min").mean().rolling(time=6, center=True).mean()
coda_sst_smooth = ds["TEMP_O2_CODA_MEAN"].where(ds["RBR_MEASUREMENT_COUNT"] == 1).resample(time="10min").mean().rolling(time=6, center=True).mean()
rbr_sss_smooth = ds["SAL_LEGATO_MEAN"].where(ds["RBR_MEASUREMENT_COUNT"] == 1).resample(time="10min").mean().rolling(time=6, center=True).mean()
rain_smooth = ds["RAIN_INTENSITY_WXT_MEAN"].resample(time="20min").mean().rolling(time=6, center=True).mean()
# Pre-calculating to keep the plotting calls clean
oxygen_smooth = ds["O2_CONC_CODA_MEAN"].where(ds["RBR_MEASUREMENT_COUNT"] == 1).resample(time="10min").mean().rolling(time=6, center=True).mean()
chl_smooth = ds["CHLOR_CYCLOPS_MEAN"].where(ds["RBR_MEASUREMENT_COUNT"] == 1).resample(time="10min").mean().rolling(time=6, center=True).mean()
rh_smooth = ds["RH_WXT_MEAN"].resample(time="10min").mean().rolling(time=6, center=True).mean()


# --- 1. Centralized Plot Config ---
# Structure: {Panel_Index: {"title": str, "vars": {name: (axis_idx, data, color, style, range, label, ylabel)}}}
# Note: axis_idx 0=ax, 1=twinx(bx), 2=twinx2(cx)
panels = {
    0: {"title": "Meteorology & Surface Conditions", "vars": {
        "aw": (0, airmar_wind_smooth, "k", ":", (0, 25, 5), "Airmar Wind", "Wind Speed (m/s)"),
        "vw": (0, vaisala_wind_smooth, "k", "-", (0, 25, 5), "Vaisala Wind", "Wind Speed (m/s)"),
        "ap": (1, airmar_pres_smooth, "0.65", ":", (980, 1030, 10), "Airmar Pressure", "Pressure (hPa)"),
        "vp": (1, vaisala_pres_smooth, "0.65", "-", (980, 1030, 10), "Vaisala Pressure", "Pressure (hPa)"),
        "wv": (2, wave_smooth, "#0A9396", "fill", (1, 6, 1), "Wave Height", "Wave Height (m)"),
        #"Uz": (0, np.abs(ds["WIND_W_GILL_MEAN"]), "k", "--", (0, 25, 5), r"|Gill U$_{z}$|", "Wind Speed (m/s)") if not plot_gill else None
    }},
    1: {"title": "Sonic Winds", "vars": {
        "Ux": (1, ds["WIND_U_GILL_MEAN"], "silver", "--", (-10, 10, 5), r"Gill U$_{x}$", "Horizontal Wind (m/s)"),
        "Uy": (1, ds["WIND_V_GILL_MEAN"], "grey", "--", (-10, 10, 5), r"Gill U$_{y}$", "Horizontal Wind (m/s)"),
        "Uz": (0, ds["WIND_W_GILL_MEAN"], "k", "-", (-1, 1, 0.5), r"Gill U$_{z}$", "Vertical Wind (m/s)"),  
    }},
    2 if plot_gill else 1: {"title": "Air-sea Interface Conditions", "vars": {
        "at": (0, airmar_air_temp_smooth, "C1", ":", (10, 25, 5), "Airmar Air Temp", "Temperature (°C)"),
        "vt": (0, vaisala_air_temp_smooth, "C1", "-", (10, 25, 5), "Vaisala Air Temp", "Temperature (°C)"),
        "ct": (0, coda_sst_smooth, "#BA0B2F", ":", (10, 25, 5), "Coda SST", "Temperature (°C)"),
        "rt": (0, rbr_sst_smooth, "#BA0B2F", "-", (10, 25, 5), "RBR SST", "Temperature (°C)"),
        "ss": (1, rbr_sss_smooth, "#4895EF", "-", (34.5, 36, 0.5), "RBR SSS", "Salinity (psu)"),
        "rn": (2, rain_smooth, "#005F73", "fill", (0, 1.5, 0.5), "Rain Intensity", "Rain (mm/hr)")
    }},
    3 if plot_gill else 2: {"title": "Biogeochemical Conditions", "vars": {
        "ox": (0, oxygen_smooth, "#003049", "-", (200, 400, 50), "Dissolved Oxygen", "Oxygen (mg/L)"),
        "ch": (1, chl_smooth, "C2", "fill", (1e-2,1e2, 1), "Chlorophyll", "Chlorophyll (mg/m³)"),
        "rh": (2, rh_smooth, "#C2B280", "-", (20, 100, 20), "Relative Humidity", "Relative Humidity (%)")
    }}
}

# --- 2. Master Loop ---
for panel_idx, panel_info in panels.items():
    main_ax = axs[panel_idx]
    # Create the twin axes on the fly and store in a list for easy access
    all_axes = [main_ax, main_ax.twinx(), main_ax.twinx()]
    all_axes[2].spines['right'].set_position(('outward', 60))
    
    handles = []
    for name, (ax_idx, data, column, linestyle, ylim, label, ylabel) in panel_info["vars"].items():
        
        cur_ax = all_axes[ax_idx]
        
        if linestyle == "fill":
            cur_ax.fill_between(data.time, data, 0, fc=column, alpha=0.15, ec=None)
            cur_ax.plot(data.time, data, color=column, lw=1, alpha=0.3, clip_on=True)
            handles.append(mpatches.Patch(color=column, alpha=0.2, label=label))
        
        else:
            cur_ax.plot(data.time, data, color=column, lw=2 if linestyle=="-" else 1, ls=linestyle, clip_on=True)
            handles.append(mlines.Line2D([], [], color=column, ls=linestyle, label=label))
	    
        if "Chlorophyll" in label:
            cur_ax.set(ylabel=ylabel, ylim=(ylim[0], ylim[1]), yscale="log")
        
        else:
            cur_ax.set(ylabel=ylabel, ylim=(ylim[0], ylim[1]), yticks=np.arange(ylim[0], ylim[1]+.1, ylim[2]))
            
        cur_ax.set_zorder(10 - 4*ax_idx); cur_ax.patch.set_visible(False)
        cur_ax.spines['top'].set_visible(False)

    main_ax.grid(axis='y', ls=':', alpha=0.4)
    main_ax.set_title(panel_info["title"], loc='left', fontsize="large", pad=20)
    main_ax.legend(handles=handles, loc='lower right', bbox_to_anchor=(1, 1), ncol=3, frameon=False, fontsize="small")
    
    if panel_idx == 1 and plot_gill:
        all_axes[2].axis('off')  # Hide the main axis for the Sonic Winds panel since it's just a legend holder
# Date Formatting
axs[2].xaxis.set(major_formatter=mdates.DateFormatter('%b %d'), major_locator=mdates.DayLocator(interval=4))

# LIVE Indicator
ts = datetime.now().strftime('%H:%M:%S UTC')
fig.text(0.99, 0.98, '●  LIVE', color='red', weight='bold', ha='right', size=12)
fig.text(0.99, 0.96, ts, color='grey', ha='right', size=8, family='monospace')

plt.savefig('/home/jedholm/share/www/html/img/waveglider_mission_2_airsea.png', transparent=True)
#plt.show()
