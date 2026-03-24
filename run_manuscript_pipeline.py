from __future__ import annotations

from calculate_manuscript_data import main as calculate_main
from figure_1_eei import main as figure_1_main
from figure_2_seasonal_masks import main as figure_2_main
from figure_3_drivers import main as figure_3_main
from figure_4_slp_dynamics import main as figure_4_main
from table_1_contributions import main as table_1_main


PIPELINE_STEPS = (
    ("Calculations", calculate_main),
    ("Figure 1", figure_1_main),
    ("Figure 2", figure_2_main),
    ("Figure 3", figure_3_main),
    ("Figure 4", figure_4_main),
    ("Table 1", table_1_main),
)


def main() -> None:
    for label, step in PIPELINE_STEPS:
        print(f"[pipeline] {label}")
        step()


if __name__ == "__main__":
    main()
