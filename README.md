# EEI trends: a process-based analysis framework

![CI](https://github.com/OrHadasAtmosphere/EEI_trends/actions/workflows/test_pkg.yml/badge.svg)

This repository contains scripts to perform analyses and generate figures ... todo

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

For instance, to generate `ceres_trends.nc` and `ceres_timeseries.nc` from the EBAF product, run
```bash
python -m obs_io.proper_ceres_seasons
```

Next, to generate figures, run (for instance)
```bash
python -m make_figs.figure_1_net_eei
```