"""Psychrometric helpers (Magnus–Tetens approximation).

Used to derive dew point from temperature/relative humidity (mock weather) and to
back-fill relative humidity from dew point when a data source omits it.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Alduchov & Eskridge (1996) coefficients, valid for -40 °C .. +50 °C.
MAGNUS_B: float = 17.625
MAGNUS_C: float = 243.04  # °C

ArrayLike = float | npt.NDArray[np.float64]


def dew_point_magnus(temperature_c: ArrayLike, relative_humidity: ArrayLike) -> ArrayLike:
    """Dew point (°C) from air temperature (°C) and relative humidity (%)."""
    rh = np.clip(np.asarray(relative_humidity, dtype=float), 1e-3, 100.0)
    t = np.asarray(temperature_c, dtype=float)
    gamma = np.log(rh / 100.0) + (MAGNUS_B * t) / (MAGNUS_C + t)
    result = MAGNUS_C * gamma / (MAGNUS_B - gamma)
    return float(result) if result.ndim == 0 else result


def relative_humidity_from_dew_point(temperature_c: ArrayLike, dew_point_c: ArrayLike) -> ArrayLike:
    """Relative humidity (%) from air temperature and dew point (both °C)."""
    t = np.asarray(temperature_c, dtype=float)
    td = np.asarray(dew_point_c, dtype=float)
    rh = 100.0 * np.exp((MAGNUS_B * td) / (MAGNUS_C + td) - (MAGNUS_B * t) / (MAGNUS_C + t))
    result = np.clip(rh, 0.0, 100.0)
    return float(result) if result.ndim == 0 else result
