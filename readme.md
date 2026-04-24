# Wave Glider Data Processor (WG-L1)

Pipeline for merging raw telemetry and AirSea logs into unified NetCDF files. This tool synchronizes multi-rate sensor data and standardizes geospatial coordinates.

## 1. System Requirements
**Environment:** Python 3.x  
**Dependencies:** `numpy`, `pandas`, `xarray`, `netCDF4`

## 2. Core Processing Logic
* **GPS Master:** Uses Airmar weather records (`waveglider-weather`) as the primary spatial reference.
* **Temporal Alignment:** Resamples all inputs to a 10-minute time vector to harmonize disparate sensor rates.
* **GPS Interpolation:** Linearly fills GPS gaps for high-frequency sensors (ADCP/AirSea) using the Airmar track as the baseline.
* **Coordinate Validation:** Enforces strict bounds and a standardized `_FillValue` for Latitude and Longitude.

## 3. Data Inputs
| Source File | Source Sensor | Parameters | Format |
| :--- | :--- | :--- | :--- |
| `weather_*.json` | Airmar 200WX | GPS, Temp, Pressure, Wind | JSON |
| `waves_*.json` | Wave Sensor | Hs, Tp, Ta, Dp | JSON |
| `adcp_*.json` | ADCP | East/North Currents | JSON |
| `AirSea_Data.csv` | RBR / Coda / Gill | Salinity, O2, Chl-a, Winds | CSV |

## 4. Metadata & Compliance
The output NetCDF applies **CF-compliant** attributes to the coordinate variables:

**Latitude / Longitude Attributes:**
* `standard_name`: `latitude` / `longitude`
* `units`: `degrees_north` / `degrees_east`
* `valid_min/max`: `[-90, 90]` / `[-180, 180]`
* `_FillValue`: `-9999.9`

## 5. Execution
Run the processor via CLI:

```bash
python wg-processor.py \
       --input <input_dir> \
       --output <output_file.nc> \
       --config <config.json>