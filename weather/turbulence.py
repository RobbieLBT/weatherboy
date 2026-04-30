"""
weather/turbulence.py

Sub-hourly wind fluctuation model using the Ornstein-Uhlenbeck process.

Rationale for OU over Von Kármán below 120 m AGL:
  - Von Kármán assumes inertial-range turbulence (valid well above the surface layer).
  - Below ~120 m, mechanical turbulence, terrain roughness, and thermal convection
    dominate. OU captures the correct statistical character (Gaussian, mean-reverting,
    exponentially correlated) without requiring a spectral decomposition.
  - METAR gust spread (gust − steady) serves as a natural proxy for σ, avoiding
    the need for external turbulence intensity tables.

Reference length scale: θ = 0.02 s⁻¹ → correlation time ~50 s, consistent with
  observed gust durations at low altitude (ESDU 83045).

Path traversal addition
-----------------------
GustSampler: online single-step OU process for use during path traversal.
  Unlike ou_series (batch), GustSampler maintains state across calls so
  the gust field is spatially correlated along the flight path.
"""

import numpy as np


def ou_series(
    mean: float,
    gust: float | None,
    n: int = 60,
    dt: float = 60.0,
    theta: float = 0.02,
    seed=None,
) -> np.ndarray:
    """
    Generate an Ornstein-Uhlenbeck wind speed time series.

    Parameters
    ----------
    mean  : mean wind speed (m/s), anchored to METAR observation
    gust  : reported gust speed (m/s); sets σ = (gust−mean)/3
            If None or ≤ mean, σ defaults to 10% of mean.
    n     : number of time steps
    dt    : step size in seconds (default 60 → 1-min resolution)
    theta : mean-reversion rate (1/s)
    seed  : optional RNG seed for reproducibility

    Returns
    -------
    (n,) array of wind speeds, clipped to ≥ 0
    """
    rng = np.random.default_rng(seed)
    sigma = (gust - mean) / 3.0 if (gust is not None and gust > mean) else max(mean * 0.10, 0.1)

    x = np.empty(n)
    x[0] = mean
    sqrt_term = np.sqrt(2.0 * theta * dt)

    for i in range(1, n):
        x[i] = (
            x[i - 1]
            + theta * (mean - x[i - 1]) * dt
            + sigma * sqrt_term * rng.standard_normal()
        )

    return np.clip(x, 0.0, None)


def gust_envelope(obs: dict, n: int = 60, dt: float = 60.0, seed=None) -> dict:
    """
    Compute gust envelope statistics for a single METAR observation interval.

    Parameters
    ----------
    obs  : obs_dict with at minimum 'wspd' and optionally 'gust'
    n    : sub-steps (default 60 → 1-hr interval at 1-min resolution)
    dt   : step size in seconds

    Returns
    -------
    {series, mean, p95, sigma}
      series : (n,) OU time series
      mean   : time-mean of series
      p95    : 95th-percentile speed (structural design value)
      sigma  : standard deviation
    """
    series = ou_series(obs["wspd"], obs.get("gust"), n=n, dt=dt, seed=seed)
    return {
        "series": series,
        "mean":   float(series.mean()),
        "p95":    float(np.percentile(series, 95)),
        "sigma":  float(series.std()),
    }


def log_wind_profile(u_ref: float, z_ref: float, z: float, z0: float = 0.03) -> float:
    """
    Logarithmic wind profile for the surface layer.

    u_ref : reference speed at z_ref (m/s), typically 10 m METAR obs
    z_ref : reference height (m)
    z     : target height (m)
    z0    : aerodynamic roughness length (m)
              0.0002 = open water
              0.03   = open terrain / short grass (default)
              0.1    = scattered obstacles
              0.5    = suburban
              1.0    = urban

    Returns speed at height z (m/s).
    """
    if z <= z0 or z_ref <= z0:
        return 0.0
    return u_ref * np.log(z / z0) / np.log(z_ref / z0)


# ── Online OU sampler for path traversal ─────────────────────────────────────

class GustSampler:
    """
    Online single-step Ornstein-Uhlenbeck gust model for path traversal.

    Unlike ou_series (which generates a batch series for a fixed obs interval),
    GustSampler maintains persistent OU state across calls so the gust perturbation
    is spatially correlated along the flight path.

    The OU state is initialized to the first mean_speed on the first call.
    Subsequent calls advance the state by one dt step, mean-reverting toward
    the current interpolated wind speed.

    Usage
    -----
    sampler = GustSampler(dt=10.0)
    for sample in path:
        instantaneous_speed = sampler.step(mean_speed, reported_gust)

    The returned speed is the instantaneous scalar wind magnitude. The caller
    is responsible for projecting it back onto the current wind direction vector.

    Parameters
    ----------
    theta : mean-reversion rate (1/s). Default 0.02 → ~50 s correlation time.
    dt    : time step in seconds. Should match path traversal dt.
    seed  : optional RNG seed.
    """

    def __init__(self, theta: float = 0.02, dt: float = 10.0, seed=None):
        self._theta     = theta
        self._dt        = dt
        self._sqrt_term = np.sqrt(2.0 * theta * dt)
        self._rng       = np.random.default_rng(seed)
        self._x: float | None = None   # OU state; None until first step

    def step(self, mean_speed: float, gust_speed: float | None) -> float:
        """
        Advance the OU process by one dt.

        Parameters
        ----------
        mean_speed : spatially interpolated mean wind speed at this sample (m/s)
        gust_speed : reported gust from nearest METAR obs (m/s), or None

        Returns
        -------
        Instantaneous wind speed perturbation (m/s), clipped to ≥ 0.
        """
        sigma = (
            (gust_speed - mean_speed) / 3.0
            if (gust_speed is not None and gust_speed > mean_speed)
            else max(mean_speed * 0.10, 0.1)
        )

        if self._x is None:
            self._x = mean_speed

        self._x = (
            self._x
            + self._theta * (mean_speed - self._x) * self._dt
            + sigma * self._sqrt_term * self._rng.standard_normal()
        )
        self._x = max(0.0, self._x)
        return self._x

    def reset(self):
        """Reset OU state (call between independent traversal runs)."""
        self._x = None
