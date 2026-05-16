from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import Rbf, RectBivariateSpline, griddata
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class InterpolationConfig:
    method: str = "cubic_spline"
    rbf_function: str = "multiquadric"
    smooth_sigma: float = 0.75


class SurfaceInterpolator:
    def __init__(self, config: InterpolationConfig | None = None):
        self.config = config or InterpolationConfig()

    def fill_missing(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        z_grid: np.ndarray,
    ) -> np.ndarray:
        values = z_grid.copy()
        missing_mask = ~np.isfinite(values)
        if not missing_mask.any():
            return values

        x_flat = x_grid[~missing_mask]
        y_flat = y_grid[~missing_mask]
        z_flat = values[~missing_mask]

        z_filled = griddata(
            points=np.column_stack([x_flat, y_flat]),
            values=z_flat,
            xi=(x_grid, y_grid),
            method="linear",
        )

        if np.isnan(z_filled).any():
            z_nn = griddata(
                points=np.column_stack([x_flat, y_flat]),
                values=z_flat,
                xi=(x_grid, y_grid),
                method="nearest",
            )
            z_filled = np.where(np.isnan(z_filled), z_nn, z_filled)
        return z_filled

    def cubic_spline(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        spline = RectBivariateSpline(x, y, z)
        return spline(x, y)

    def rbf(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        z_grid: np.ndarray,
    ) -> np.ndarray:
        x_flat = x_grid.ravel()
        y_flat = y_grid.ravel()
        z_flat = z_grid.ravel()
        model = Rbf(x_flat, y_flat, z_flat, function=self.config.rbf_function)
        return model(x_grid, y_grid)

    def smooth(self, z_grid: np.ndarray) -> np.ndarray:
        clipped = np.clip(z_grid, 1e-4, 5.0)
        return gaussian_filter(clipped, sigma=self.config.smooth_sigma)

    def interpolate_surface(
        self,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        z_grid: np.ndarray,
    ) -> np.ndarray:
        x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="xy")
        filled = self.fill_missing(x_grid, y_grid, z_grid)
        if self.config.method == "rbf":
            interpolated = self.rbf(x_grid, y_grid, filled)
        else:
            interpolated = self.cubic_spline(y_axis, x_axis, filled).astype(float)
        return self.smooth(interpolated)
