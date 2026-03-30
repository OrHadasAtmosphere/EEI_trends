from __future__ import annotations

import argparse
import numpy as np
import xarray as xr

from calculate_manuscript_data import (
    CONTRIBUTIONS_OUTPUT_FILE,
    OUTPUT_DIR,
    OVERLEAF_FIGURES_DIR,
    SEASONS,
    save_text_outputs,
    ensure_manuscript_outputs,
)

OREDER = [-2, -4, -3, 2, 3, 0, 1, -1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print Table 1 as LaTeX. By default, statistically significant trends "
            "are bolded instead of showing the 95% confidence interval."
        )
    )
    parser.add_argument(
        "--show-ci",
        action="store_true",
        help="Show trend ± 95%% confidence interval instead of bolding significant trends.",
    )
    return parser.parse_args()


def format_cell(
    trend: float,
    ci95: float,
    contribution_percent: float,
    *,
    show_ci: bool,
) -> str:
    if not (
        np.isfinite(trend) and np.isfinite(ci95) and np.isfinite(contribution_percent)
    ):
        return "---"

    trend_decade = trend * 10.0
    ci95_decade = ci95 * 10.0

    if show_ci:
        return f"${trend_decade:.1f} \\pm {ci95_decade:.1f}\\;({contribution_percent:.0f}\\%)$"

    trend_text = f"{trend_decade:.2f}"
    if abs(trend) > ci95:
        trend_text = rf"\mathbf{{{trend_text}}}"
    return f"${trend_text}\\;({contribution_percent:.1f}\\%)$"


def format_region_label(ds: xr.Dataset, region: str) -> str:
    mean_seasonal_area_percent = (
        float(ds["seasonal_area_fraction"].sel(region=region).mean("season")) * 100.0
    )
    return f"{region} ({mean_seasonal_area_percent:.0f}\\%)"


def format_mask_label(ds: xr.Dataset, name: str, area_var: str) -> str:
    mean_seasonal_area_percent = float(ds[area_var].mean("season")) * 100.0
    return f"{name} ({mean_seasonal_area_percent:.0f}\\%)"


def main() -> None:
    args = parse_args()
    ensure_manuscript_outputs(force=False)
    ds = xr.open_dataset(CONTRIBUTIONS_OUTPUT_FILE)

    lines = [
        "\\begin{tabular}{llllll}",
        "\\hline",
        "Region (mean seasonal area) & Annual & DJF & MAM & JJA & SON \\\\",
        "\\hline",
    ]
    regions = ds["region"].values
    for o in OREDER:
        region = regions[o]
        annual_cell = format_cell(
            float(ds["annual_region_trend"].sel(region=region)),
            float(ds["annual_region_ci95"].sel(region=region)),
            float(ds["annual_contribution_percent"].sel(region=region)),
            show_ci=args.show_ci,
        )
        seasonal_cells = [
            format_cell(
                float(ds["seasonal_region_trend"].sel(season=season, region=region)),
                float(ds["seasonal_region_ci95"].sel(season=season, region=region)),
                float(
                    ds["seasonal_contribution_percent"].sel(
                        season=season,
                        region=region,
                    )
                ),
                show_ci=args.show_ci,
            )
            for season in SEASONS
        ]
        cells = " & ".join([annual_cell, *seasonal_cells])
        lines.append(f"{format_region_label(ds, region)} & {cells} \\\\")
    overall_annual = format_cell(
        float(ds["annual_global_trend"]),
        float(ds["annual_global_ci95"]),
        100.0,
        show_ci=args.show_ci,
    )
    overall_seasonal = [
        format_cell(
            float(ds["seasonal_global_trend"].sel(season=season)),
            float(ds["seasonal_global_ci95"].sel(season=season)),
            float(ds["seasonal_global_contribution_percent"].sel(season=season)),
            show_ci=args.show_ci,
        )
        for season in SEASONS
    ]
    overall_cells = " & ".join([overall_annual, *overall_seasonal])
    lines.append(f"Overall & {overall_cells} \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    latex = "\n".join(lines)
    print(latex)

    local_path, overleaf_path = save_text_outputs(
        latex + "\n",
        OUTPUT_DIR / "table_1_contributions.tex",
        OVERLEAF_FIGURES_DIR / "table_1_contributions.tex",
    )
    print(local_path)
    print(overleaf_path)


if __name__ == "__main__":
    main()
