# EEI trends: a process-based analysis framework

![CI](https://github.com/OrHadasAtmosphere/EEI_trends/actions/workflows/test_pkg.yml/badge.svg)

This repository supports a decomposition of observed EEI trends into dynamical regimes and a process-based analysis of their origin.

## setup with pixi

We use [Pixi](https://pixi.sh/) to manage Python packages and their dependencies. Pixi handles both conda packages (like `esmpy`) and pip packages in a single environment, eliminating the need for separate package managers.

First install pixi if you don't already have it:

```bash
# macOS/Linux
curl -fsSL https://pixi.sh/install.sh | bash

# Windows (PowerShell)
iwr -useb https://pixi.sh/install.ps1 | iex
```

Then set up the environment with

```bash
pixi install

pixi run setup
```

Once the items described in [this README](raw_data/README.md) are downloaded to `raw_data/`, you can run processing, analysis, and plotting with commands grouped into sub tasks or by specific task, e.g.:

```bash
pixi run process-data # downloads ERA5 data, processes to netcdfs with regimes
pixi run analysis # performs further computations on regimes and pre-processed data
pixi run main-plots # makes plots for the main manuscript
pixi run all # runs all prior commands, plus supplemental figures and sensitivity analyses
```

More commands, defined in `pixi.toml` can be also be displayed with `pixi run`.

## Windows users:

Our dependency on `xesmf` unfortunately creates problems for Windows users causing our dependencies to be incompatible with your operating system.

For workarounds see [this `xesmf` docs note](https://xesmf.readthedocs.io/en/stable/installation.html#notes-for-windows-users) and note the [issue](https://github.com/OrHadasAtmosphere/EEI_trends/issues/18) in our repo.