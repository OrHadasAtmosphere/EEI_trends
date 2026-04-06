from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import xarray as xr


SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(SCRIPT_DIR / ".cache"))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from calculate_manuscript_data import (  # noqa: E402
    CERES_OUTPUT_FILE,
    CONTRIBUTIONS_OUTPUT_FILE,
    MASKS_OUTPUT_FILE,
    ensure_manuscript_outputs,
)


AREA_ATOL = 1.0e-6
CONTRIBUTION_ATOL = 5.0e-4


class TestManuscriptClosure(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ensure_manuscript_outputs(force=False)
        cls.ceres = xr.open_dataset(CERES_OUTPUT_FILE)
        cls.contributions = xr.open_dataset(CONTRIBUTIONS_OUTPUT_FILE)
        cls.masks = xr.open_dataset(MASKS_OUTPUT_FILE)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.ceres.close()
        cls.contributions.close()
        cls.masks.close()

    def test_seasonal_area_partition_closes_to_global_area(self) -> None:
        seasonal_area_total = (
            self.contributions["seasonal_area_fraction"].sum("region")
            + self.contributions["desert_area_fraction"]
        )
        np.testing.assert_allclose(
            seasonal_area_total.values,
            np.ones(seasonal_area_total.size, dtype=np.float64),
            atol=AREA_ATOL,
        )

    def test_region_masks_and_deserts_are_disjoint_and_exhaustive(self) -> None:
        region_assignments = self.masks["region_mask"].astype(np.int16).sum("region")
        total_assignments = region_assignments + self.masks[
            "positive_omega_land_mask"
        ].astype(np.int16)

        self.assertEqual(int((region_assignments > 1).sum()), 0)
        self.assertEqual(int((total_assignments > 1).sum()), 0)
        self.assertEqual(int((total_assignments == 0).sum()), 0)

    def test_seasonal_contributions_close_to_seasonal_global_total(self) -> None:
        seasonal_contribution_total = (
            self.contributions["seasonal_contribution_percent"].sum("region")
            + self.contributions["desert_contribution_percent"]
        )
        np.testing.assert_allclose(
            seasonal_contribution_total.values,
            self.contributions["seasonal_global_contribution_percent"].values,
            atol=CONTRIBUTION_ATOL,
        )

    def test_annual_region_contributions_equal_sum_of_seasonal_contributions(self) -> None:
        seasonal_region_total = self.contributions["seasonal_contribution_percent"].sum(
            "season"
        )
        np.testing.assert_allclose(
            self.contributions["annual_contribution_percent"].values,
            seasonal_region_total.values,
            atol=CONTRIBUTION_ATOL,
        )

    def test_annual_desert_contribution_equals_sum_of_seasonal_desert_contributions(
        self,
    ) -> None:
        np.testing.assert_allclose(
            float(self.contributions["annual_desert_contribution_percent"]),
            float(self.contributions["desert_contribution_percent"].sum("season")),
            atol=CONTRIBUTION_ATOL,
        )

    def test_annual_partition_matches_sum_of_seasonal_global_contributions(
        self,
    ) -> None:
        annual_partition_total = float(
            self.contributions["annual_contribution_percent"].sum("region")
            + self.contributions["annual_desert_contribution_percent"]
        )
        seasonal_global_total = float(
            self.contributions["seasonal_global_contribution_percent"].sum("season")
        )
        np.testing.assert_allclose(
            annual_partition_total,
            seasonal_global_total,
            atol=CONTRIBUTION_ATOL,
        )

    def test_seasonal_global_contributions_sum_to_100_percent(self) -> None:
        np.testing.assert_allclose(
            float(self.contributions["seasonal_global_contribution_percent"].sum()),
            100.0,
            atol=CONTRIBUTION_ATOL,
        )

    def test_annual_region_and_desert_contributions_sum_to_100_percent(self) -> None:
        np.testing.assert_allclose(
            float(
                self.contributions["annual_contribution_percent"].sum("region")
            ),
            100.0,
            atol=CONTRIBUTION_ATOL,
        )

    def test_direct_and_reconstructed_annual_global_trends_match(self) -> None:
        np.testing.assert_allclose(
            float(self.ceres["all_sky_global_net_annual_trend"]),
            float(self.contributions["annual_global_trend"]),
            atol=CONTRIBUTION_ATOL,
        )


"""""
1. All grid points are assigned to exactly one region. No tolerence.
2. The sum of areas of all masks is about the area of Earth. Up to 1% accuracy.
3. Each season sums up to the global contribution. No tolerence.
4. The total contribution of the seasons sums up to the annual contribution. No tolerence.
"""""

if __name__ == "__main__":
    unittest.main()
