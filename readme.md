# Wave Glider Data Processor (WG-L1)

Pipeline for merging raw telemetry and AirSea logs into CF-compliant NetCDF files. This tool synchronizes multi-rate sensor data and standardizes geospatial metadata.

## 1. System Requirements
**Environment:** Python 3.x  
**Dependencies:** `numpy`, `pandas`, `xarray`, `netCDF4`

## 2. Core Processing Logic
* **GPS Master:** Uses Airmar weather records as the primary spatial reference.
* **Temporal Alignment:** Resamples all inputs to a unified 10-minute time vector.
* **Metadata Injection:** Automatically generates dimensionless `SENSOR_` variables and populates CF-standard attributes.

## 3. Data Inputs
| Source File | Source Sensor | Parameters | Format |
| :--- | :--- | :--- | :--- |
| `weather_*.json` | Airmar 200WX | GPS, Temp, Pressure, Wind | JSON |
| `waves_*.json` | Wave Sensor | Hs, Tp, Ta, Dp | JSON |
| `adcp_*.json` | ADCP | U/V Currents (Bin-mapped) | JSON |
| `AirSea_Data.csv` | RBR / Coda / Gill | Salinity, O2, Chl-a, Winds | CSV |

## 4. Required Configuration (`config.json`)

The script requires a configuration file defining:

* `mission_config`: `start_time`, `end_time`, `mission_name`.

* `sensors`: Sensor-specific metadata (serial numbers, models).

* `global_attributes`: Project-level metadata (PI, institution).

### Coordinate Attributes
* **Latitude/Longitude:** `degrees_north` / `degrees_east`
* **Valid Range:** `[-90, 90]` / `[-180, 180]`
* **Fill Value:** `-9999.9`

## 5. Execution
Run the processor via CLI:

```bash
python wg-processor.py \
       --input <input_dir> \
       --output <output_file.nc> \
       --config <config.json>