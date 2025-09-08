#  

The Satellite Precipitation Estimation Evaluation Dataset (SPEED) package is a utility package for collocating GPM satellite observations with various reference datasets. The package has been developed to create the [SatRain](github.com/ipwgml/satrain) benchmark dataset and is actively used for validating [GPROF-NN](github.com/simonpf/gprof_nn) retrievals.

## Installation

### Using conda (recommended)

```bash
# Create and activate environment
conda env create -f environment.yml
conda activate speed

# Install package
pip install -e .
```

### Using pip

```bash
pip install speed
```

For development:
```bash
pip install -e ".[dev]"
```

## Usage

SPEED provides a command-line interface for data extraction and processing:

```bash
# Extract collocations between input and reference data
speed extract_data <input_data> <reference_data> <output_folder> <year> <month> [days...]

# Extract training data from collocations
speed extract_training_data_spatial <collocation_path> <output_folder>
speed extract_training_data_tabular <collocation_path> <output_folder>

# Extract evaluation data
speed extract_evaluation_data <collocation_path> <output_folder>
```

For detailed usage information:
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

## Development

Run tests:
```bash
pytest
```

For more development information, see [CLAUDE.md](CLAUDE.md).

## Status

The dataset is currently in version 0.1.0. If you have suggestions please open a new issue.

## License

MIT License - see LICENSE file for details.

