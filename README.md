# EEI trends: a process-based analysis framework

![CI](https://github.com/OrHadasAtmosphere/EEI_trends/actions/workflows/test_pkg.yml/badge.svg)

This repository supports a decomposition of observed EEI trends into dynamical regimes and a process-based analysis of their origin.

## Environment Setup with conda/micromamba

Using a package manager like `micromamba`, `conda`, `mamba`, `miniconda`, or another serpentine permutation ensures reproducibility, forward compatibility, and ease of collaboration. We provide an `environment.yaml` with fixed dependencies, configurable as follows:

```bash
micromamba env create -f environment.yaml
micromamba activate eei-env
```

## Package installation and running scripts

The code is configured as a Python package to facilitate cross-file dependencies and a centralization of common tools. After activating the Python environment, install the package:

```bash
pip install .
```

Note: for development, you probably want to install the package in editor mode:
```bash
pip install -e .
```

To run scripts using the modular structure used here, use the syntax:
```bash 
python -m module.script
```

Once the items described in [this README](raw_data/README.md) are downloaded to `raw_data/`, `pipeline.sh` executes all necessary commands to analyze data and to generate figures. You can run this with

```bash
bash pipeline.sh
```

or simply execute the commands contained in the file at your leisure.

## For Windows users:

Our dependency on `xesmf` unfortunately creates problems for Windows users causing our dependencies to be incompatible with your operating system.

For workarounds see [this `xesmf` docs note](https://xesmf.readthedocs.io/en/stable/installation.html#notes-for-windows-users) and note the [issue](https://github.com/OrHadasAtmosphere/EEI_trends/issues/18) in our repo.