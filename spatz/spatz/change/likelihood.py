"""Likelihood estimation for phase noise."""
# SpaTZ, SPAtioTemporal Selection of Scatterers
#
# Copyright (c) 2024  Andreas Piter
# (Institute of Photogrammetry and GeoInformation, Leibniz University Hannover)
# Contact: piter@ipi.uni-hannover.de
#
# This software was developed within the context of the PhD thesis of Andreas Piter.
# The software is not yet licensed and used for internal development only.
from typing import Union
import numpy as np


def marginalPDFInterferometricPhase(*,
                                    coherence: float,
                                    phase: Union[np.ndarray, float],
                                    phase_0: float) -> Union[np.ndarray, float]:
    """Compute the marginal PDF of the interferometric phase.

    Example for plotting:
    xval = np.linspace(-np.pi, np.pi, 100)
    yval = marginalPDFInterferometricPhase(coherence=0.8, phase=xval, phase_0=0)
    plt.plot(xval, yval)

    Parameters
    ----------
    coherence : float
        Temporal coherence of the interferogram.
    phase : Union[np.ndarray, float]
        Interferometric phase. Can be either a single phase value or an array of phase values.
    phase_0 : float
        Expected value of the phase (often assumed to be zero).

    Returns
    -------
    pdf : Union[np.ndarray, float]
        Marginal PDF of the interferometric phase. Can be either a single PDF value or an array of PDF values.
    """
    delta_phase = phase - phase_0
    beta = 1 - coherence ** 2 * np.cos(delta_phase) ** 2

    if np.any(beta <= 0):  # avoid numerical issues
        beta[beta == 0] = np.nan

    pdf = ((1 - coherence ** 2) / (2 * np.pi) *
           1 / beta *
           (1 + (coherence * np.cos(delta_phase) * np.arccos(- coherence * np.cos(delta_phase)))
            / (beta ** 0.5))
           )
    return pdf


def logLikelihoodInterferometricPhase(coherence: float, ifg_res: np.ndarray, phase_0: float) -> float:
    """Compute log-likelihood of the interferometric phase for Likelihood Ratio Test (LRT).

    Parameters
    ----------
    coherence : float
        Temporal coherence of the interferogram.
    ifg_res : np.ndarray
        Interferometric phase residuals.
    phase_0 : float
        Expected value of the phase.

    Returns
    -------
    loglikelhd : float
        Log-likelihood of the interferometric phase.
    """
    # for each interferogram
    f = marginalPDFInterferometricPhase(coherence=coherence, phase=ifg_res, phase_0=phase_0)

    # logarithm of each ifg
    logf = np.log(f)

    # sum of all logrithms
    loglikelhd = np.sum(logf)
    return loglikelhd
