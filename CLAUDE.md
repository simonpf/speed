# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SPEED (Satellite Precipitation Estimation Evaluation Dataset) is a benchmark dataset for precipitation estimation algorithms developed by the International Precipitation Working Group (IPWG). The repository contains code for dataset generation, access, and usage.

## Common Commands

### Installation and Environment Setup
```bash
# Create conda environment (recommended)
conda env create -f environment.yml
conda activate speed

# Install package in development mode
pip install -e .
```

### CLI Usage
The main entry point is the `speed` command which provides several subcommands:

```bash
# Extract collocations between input and reference data
speed extract_data <input_data> <reference_data> <output_folder> <year> <month> [days...] [-n N_PROCESSES]

# Extract spatial training data from collocations
speed extract_training_data_spatial <collocation_path> <output_folder> [--overlap OVERLAP] [--size SIZE] [--include_geo] [--include_geo_ir] [-n N_PROCESSES]

# Extract tabular training data from collocations
speed extract_training_data_tabular <collocation_path> <output_folder> [--include_geo] [--include_geo_ir] [--subsample FRACTION]

# Extract evaluation data
speed extract_evaluation_data <collocation_path> <output_folder> [--include_geo] [--include_geo_ir] [--glob_pattern PATTERN]

# Extract specific observation types
speed extract_cpcir_obs <args>
speed extract_goes_obs <args>
speed extract_himawari_obs <args>
speed add_ancillary_data <args>
```

### Testing
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest test/data/test_utils.py

# Run tests matching pattern
python -m pytest -k "test_pattern"
```

### Shell Scripts
The `scripts/` directory contains extraction scripts for various data combinations:
```bash
# Example: Extract MRMS-GMI collocations
./scripts/extract_mrms_collocations_gmi.sh <year> <month>
```

## Code Architecture

### Core Structure
- **`speed/`** - Main package directory
  - **`cli.py`** - Command-line interface using Click
  - **`data/`** - Data handling modules for different sources
    - **`input.py`** - Input dataset management
    - **`reference.py`** - Reference dataset management
    - **`utils.py`** - Data processing utilities
    - **Individual sensor modules**: `gpm.py`, `seviri.py`, `mrms.py`, etc.
  - **`grids/`** - Grid handling functionality
  - **`logging.py`** - Logging configuration
  - **`plotting.py`** - Visualization utilities

### Data Sources
The codebase supports multiple satellite and ground-based precipitation data sources:
- **Satellite**: GPM (GMI, DPR), GOES, Himawari, SEVIRI, AMSR2, MHS, ATMS
- **Ground-based**: MRMS, GPM Ground Validation, Wegener Net, AMEDAS, KMA
- **Reanalysis**: Various ancillary datasets

### Key Design Patterns
- **Dataset abstraction**: Input and reference datasets implement common interfaces
- **Collocation workflow**: Extract spatial and temporal matches between datasets
- **Parallel processing**: Built-in multiprocessing support for data extraction
- **Modular data sources**: Each data type (GPM, MRMS, etc.) has its own module

### Dependencies
- **pansat**: Main dependency for satellite data access (custom fork)
- **xarray**: Primary data structure for N-dimensional arrays
- **click**: Command-line interface framework
- **pytest**: Testing framework
- **Standard scientific stack**: numpy, scipy, matplotlib, cartopy

### Testing
Tests are organized under `test/` with fixtures in `conftest.py` that provide sample data granules and matches. Many tests require the `PANSAT_PASSWORD` environment variable for data access.