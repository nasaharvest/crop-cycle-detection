"""
NDVI preprocessing utilities for the crop cycle model.

Provides:
  - local_regression   : locally weighted linear regression for outlier detection
  - whittaker_smooth   : penalised least-squares smoother for NDVI time series
  - setup_logging      : convenience logger configuration for notebooks / scripts
"""

import logging

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def radial_kernel(x0: np.ndarray, X: np.ndarray, k: float) -> np.ndarray:
    """
    Radial (Gaussian) kernel for local regression.

    Parameters
    ----------
    x0 : np.ndarray
        Query point (with bias term).
    X : np.ndarray
        Input points (with bias term).
    k : float
        Bandwidth parameter.

    Returns
    -------
    np.ndarray
        Per-point weights relative to x0.
    """
    return np.exp(-np.sum((X - x0) ** 2, axis=1) / (2 * k * k))


def local_regression(x0: float, X: list, Y: list, k: float) -> float:
    """
    Locally weighted linear regression at a single query point.

    Parameters
    ----------
    x0 : float
        Query point to predict.
    X : list
        Independent variable values (e.g. days since start).
    Y : list
        Dependent variable values (e.g. NDVI).
    k : float
        Radial kernel bandwidth.

    Returns
    -------
    float
        Predicted Y at x0.
    """
    x0 = np.r_[1, x0]
    X = np.c_[np.ones(len(X)), X]
    weights = radial_kernel(x0, X, k)
    Xw = X.T * weights
    beta = np.linalg.pinv(Xw @ X) @ Xw @ Y
    return x0 @ beta


def whittaker_smooth(
    y: np.ndarray,
    lam: float = 6_000,
    differences: int = 2,
) -> np.ndarray:
    """
    Whittaker smoother — penalised least-squares smoothing of a 1-D signal.

    Solves:  min_z  ||y - z||^2 + λ ||D^d z||^2

    where D is the first-order difference operator and d = ``differences``.
    Larger ``lam`` yields a smoother result.

    Parameters
    ----------
    y : np.ndarray
        1-D observed values (e.g. daily NDVI after PCHIP interpolation).
    lam : float, optional
        Smoothing penalty λ. Default 6 000.
    differences : int, optional
        Order of the finite-difference penalty. Default 2.

    Returns
    -------
    np.ndarray
        Smoothed signal, same length as ``y``.

    Raises
    ------
    ValueError
        If ``y`` is empty, not 1-D, or ``differences < 1``.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("whittaker_smooth expects a 1-D array for `y`.")
    if y.size == 0:
        raise ValueError("Input array `y` must not be empty.")
    if differences < 1:
        raise ValueError("`differences` must be >= 1.")

    n = y.size
    E = sparse.eye(n, format="csr")
    D = E[1:, :] - E[:-1, :]
    for _ in range(differences - 1):
        D = D[1:, :] - D[:-1, :]

    A = E + lam * (D.T @ D)
    return np.asarray(spsolve(A, y), dtype=float)


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure the root logger with a standard format.

    Call once at the top of a script or notebook to enable timestamped output.

    Parameters
    ----------
    level : int, optional
        Logging level (e.g. ``logging.INFO``, ``logging.DEBUG``).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    )
