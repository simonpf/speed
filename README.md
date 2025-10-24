# SPEED

Collocating satellite observations and precipitation reference data for faster evaluation of precipitation retrievals.

The Satellite Precipitation Estimation Evaluation Data (SPEED) package is a utility package for collocating GPM satellite observations with various reference datasets. The package has been developed to create the [SatRain](github.com/ipwgml/satrain) benchmark dataset and is actively used for validating [GPROF-NN](github.com/simonpf/gprof_nn) retrievals.

## Installation

### Using conda (recommended)

```bash
# Create and activate environment
conda env create -f environment.yml
conda activate speed

# Install package
pip install -e .
```

## Usage

SPEED provides a command-line interface for data extraction and processing:

```bash
# Extract collocations between input and reference data
speed extract_data <input_data> <reference_data> <output_folder> <year> <month> [days...]
```

For additional usage information:
```bash
speed --help
```

## Supported Data Sources

### Satellite Data
- **GPM**: GMI, DPR
- **GOES**: Geostationary IR observations
- **Himawari**: Japanese geostationary satellite
- **SEVIRI**: Meteosat observations
- **AMSR2**: Microwave radiometer
- **MHS/ATMS**: Microwave sounders

### Ground-based Data
- **MRMS**: Multi-Radar/Multi-Sensor system
- **GPM Ground Validation**: Ground-based precipitation measurements
- **Wegener Net**: High-resolution station network
- **AMEDAS**: Japan Meteorological Agency stations
- **KMA**: Korea Meteorological Administration

## License

MIT License - see LICENSE file for details.

