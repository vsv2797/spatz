"""Change detection using coherence matrix."""
# SpaTZ, SPAtioTemporal Selection of Scatterers
#
# Copyright (c) 2024  Andreas Piter
# (Institute of Photogrammetry and GeoInformation, Leibniz University Hannover)
# Contact: piter@ipi.uni-hannover.de
#
# This software was developed within the context of the PhD thesis of Andreas Piter.
# The software is not yet licensed and used for internal development only.
import logging
import numpy as np
from matplotlib import pyplot as plt

from mintpy.utils import ptime


def runCoherenceMatrix(*,
                       slc: np.ndarray,
                       mask_cand: np.ndarray,
                       logger: logging.Logger,
                       model_coherence: float = 0.5,
                       wdw_size: int = 9) -> np.ndarray:
    """Run change detection based on phase noise estimation with a spatial displacement model.

    Steps:
    1. Precompute the likelihood terms which are the same for each pixel.
    2. Extract the SLC time series of the TCS and its spatial neighbourhood within the window.
    3. Estimate the coherence matrix for each pixel by incorporating the spatial neighbourhood. Pixels at the boundary
       of the image are skipped as they cannot be processed due to the window size.
    4. Run the change detection algorithm for each pixel.

    Parameters
    ----------
    slc : np.ndarray
        SLC stack (number of images x length x width).
    mask_cand : np.ndarray
        Binary mask image indicating the location of TCS candidates that shall be examined (length x width).
    logger : logging.Logger
        Logger object.
    model_coherence : float
        Model coherence value used to construct a model for the coherence matrix.
    wdw_size : int
        Window size for the coherence matrix estimation from the spatial neighbourhood of a pixel.

    Returns
    -------
    change_time_map : np.ndarray
        Image with the estimated change time index for each pixel (length x width). Change time is within interval
        [0, number of images - 1]. The change time index is np.nan for pixels that could not be processed.
    """
    logger.info(msg="COHERENCE MATRIX Change detection.")

    wdw_hfs = int(wdw_size // 2)
    num_cand = (np.where(mask_cand)[0]).shape[0]
    coord_tcs_cand = np.where(mask_cand)
    coord_x_cand = coord_tcs_cand[0]
    coord_y_cand = coord_tcs_cand[1]
    step_time = np.zeros(num_cand, dtype=np.int16)

    """The number of samples is not the same for pixels at the image border, but those pixels are discarded anyway"""
    num_indep_samples = (wdw_hfs * 2 + 1) ** 2

    term1, weight_mat = precomputeLikelihoodTerms(
        num_images=slc.shape[0],
        model_coherence=model_coherence,
        num_indep_samples=num_indep_samples
    )

    prog_bar = ptime.progressBar(maxValue=num_cand)
    for idx in range(num_cand):
        x = coord_x_cand[idx]
        y = coord_y_cand[idx]

        y1 = y - wdw_hfs if y - wdw_hfs >= 0 else 0
        y2 = y + wdw_hfs + 1 if y + wdw_hfs + 1 <= slc.shape[2] else slc.shape[2]
        x1 = x - wdw_hfs if x - wdw_hfs >= 0 else 0
        x2 = x + wdw_hfs + 1 if x + wdw_hfs + 1 <= slc.shape[1] else slc.shape[1]
        slc_wdw = slc[:, x1:x2, y1:y2]

        if (slc_wdw.shape[1] != (wdw_hfs * 2 + 1)) | (slc_wdw.shape[2] != (wdw_hfs * 2 + 1)):
            # skip problems at the boundary of the image. It is an issue in the unwrapping approach. So skip it here too
            step_time[idx] = -1
            continue

        coh_mat = np.abs(np.corrcoef(slc_wdw.reshape(slc_wdw.shape[0], -1)))

        score = singleCDPrecomputed(
            coh_mat=coh_mat,
            num_indep_samples=num_indep_samples,
            term1=term1,
            weight_mat=weight_mat
        )

        step_time[idx] = np.nanargmin(score) + 1
        prog_bar.update(idx + 1, every=np.int16(num_cand / 200),
                        suffix='{}/{} candidates processed.'.format(idx + 1, num_cand))
    prog_bar.close()
    logger.info(f"{num_cand} TCS candidates processed.")

    img_size = slc.shape[1:]
    change_time_map = np.zeros(img_size, dtype=np.int16)
    change_time_map[mask_cand] = step_time
    return change_time_map


def singleCDPrecomputed(*,
                        coh_mat: np.ndarray,
                        num_indep_samples: int,
                        term1: np.ndarray,
                        weight_mat: np.ndarray) -> np.ndarray:
    """Change detection with coherence matrix model assuming a single change point in the data.

    This method is a precomputed version of the function 'singleCD(...)'.
    The terms for the likelihood ratio test were precomputed. This includes term1 and the following part of term2:

    LRT = Term1 - Term2   # Monti-Guarnieri et al. (2018) write: LRT = Term1 + Term2
    Term1 = N_S * log(det(C_change) / det(C_no-change))
    Term2 = N_S * trace(W * C)
    W = inv(C_no-change) - inv(C_change)  # Monti-Guarnieri et al. (2018) write: W = inv(C_change) - inv(C_no-change)

    Parameters
    ----------
    coh_mat : np.ndarray
        Coherence matrix (number of images x number of images).
    num_indep_samples : int
        Number of independent samples (neighbouring pixels) used to estimate the coherence matrix.
    term1 : np.ndarray
        Precomputed term1 for the likelihood ratio test.
    weight_mat : np.ndarray
        Precomputed weight matrix for the likelihood ratio test.

    Returns
    -------
    score : np.ndarray
        Score of the likelihood ratio for each step time.
    """
    num_images = coh_mat.shape[0]
    score = np.zeros((num_images - 1,))

    for step_time_idx in range(1, num_images):
        term2 = num_indep_samples * np.trace(np.matmul(weight_mat[step_time_idx - 1], coh_mat))
        score[step_time_idx - 1] = term1[step_time_idx - 1] - term2

    return score


def singleCD(*,
             coh_mat: np.ndarray,
             model_coherence: float,
             num_indep_samples: int,
             show_plots: bool = False) -> np.ndarray:
    """Change detection with coherence matrix model assuming a single change point in the data.

    Calling this function multiple times is slow, as the coherence matrix has to be inverted for each call. Use the
    precomputed version 'singleCDPrecomputed(...)' instead.

    The Likelihood ratio test (LRT) is defined in Monti-Guarnieri et al. (2018) in equation (14). However, it differs
    from the implementation in Manzoni et al. (2021) in RSE, equation (9). The likelihood ratio test is defined as:

    LRT = Term1 - Term2   # Monti-Guarnieri et al. (2018) write: LRT = Term1 + Term2
    Term1 = N_S * log(det(C_change) / det(C_no-change))
    Term2 = N_S * trace(W * C)
    W = inv(C_no-change) - inv(C_change)  # Monti-Guarnieri et al. (2018) write: W = inv(C_change) - inv(C_no-change)

    Parameters
    ----------
    coh_mat : np.ndarray
        Coherence matrix (number of images x number of images).
    model_coherence : float
        Model coherence value.
    num_indep_samples : int
        Number of independent samples (neighbouring pixels) used to estimate the coherence matrix.
    show_plots : bool
        Show plots for debugging.

    Returns
    -------
    np.ndarray
        Score of the likelihood ratio for each step time.
    """
    num_images = coh_mat.shape[0]
    score = np.zeros((num_images - 1,))
    score_term1 = np.zeros((num_images - 1,))
    score_term2 = np.zeros((num_images - 1,))

    num_indep_samples = num_indep_samples
    num_images = coh_mat.shape[0]
    coh_mat_stable = np.ones((num_images, num_images), dtype=np.float64) * model_coherence
    coh_mat_stable[np.diag_indices(num_images)] = 1

    # pre-compute for speed
    inv_coh_mat_stable = np.linalg.inv(coh_mat_stable)
    det_coh_mat_stable = np.linalg.det(coh_mat_stable)

    for step_time_idx in range(1, num_images):
        coh_mat_change = coh_mat_stable.copy()
        coh_mat_change[:step_time_idx, step_time_idx:] = 0
        coh_mat_change[step_time_idx:, :step_time_idx] = 0

        term1 = num_indep_samples * np.log(
            np.linalg.det(coh_mat_change) / det_coh_mat_stable
        )
        term2 = num_indep_samples * np.trace(
            np.matmul((inv_coh_mat_stable - np.linalg.inv(coh_mat_change)), coh_mat)
        )
        score[step_time_idx - 1] = term1 - term2  # We use 'minus' as in Manzoni et al. (2021) RSE, eq. 9
        score_term1[step_time_idx - 1] = term1
        score_term2[step_time_idx - 1] = term2

    if show_plots:
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
        ax1.imshow(coh_mat, vmin=0, vmax=1., cmap=plt.cm.gray)
        xsteps = np.linspace(0.5, num_images - 1.5, num=num_images - 1)
        ax2.plot(xsteps, score, '.-k', label="GLRT score")
        ax2.plot(xsteps, score_term1, '.-b', label="GLRT score - term 1")
        ax2.plot(xsteps, score_term2, '.-r', label="GLRT score - term 2")
        ax2.legend()
        ax2.set_xlabel("Acquisition no.")
        ax2.set_ylabel("Score")

    if show_plots:
        def plotChangeFromScore(*, score: np.ndarray, col: str):
            ax1.plot([np.argmin(score) + 0.5, np.argmin(score) + 0.5], [0, num_images - 1], '-' + col, linewidth=0.5)
            ax1.plot([0, num_images - 1], [np.argmin(score) + 0.5, np.argmin(score) + 0.5], '-' + col, linewidth=0.5)
            ax2.plot([np.argmin(score) + 0.5, np.argmin(score) + 0.5], [np.min(score), np.max(score)], '-' + col,
                     linewidth=0.5)

        plotChangeFromScore(score=score, col='k')
        plotChangeFromScore(score=score_term2, col='r')

    return score


def precomputeLikelihoodTerms(*,
                              num_images: int,
                              model_coherence: float,
                              num_indep_samples: int) -> (np.ndarray, np.ndarray):
    """Precompute terms of likelihood for the coherence matrix change detection.

    Create lookup table with terms of likelihood for all possible change steps.
    The likelihood ratio test (LRT) is defined in Monti-Guarnieri et al. (2018) in equation (14). However, it differs
    from the implementation in Manzoni et al. (2021) in RSE, equation (9). The likelihood ratio test is defined as:

    LRT = Term1 - Term2   # Monti-Guarnieri et al. (2018) write: LRT = Term1 + Term2
    Term1 = N_S * log(det(C_change) / det(C_no-change))
    Term2 = N_S * trace(W * C)
    W = inv(C_no-change) - inv(C_change)  # Monti-Guarnieri et al. (2018) write: W = inv(C_change) - inv(C_no-change)

    Parameters
    ----------
    num_images : int
        Number of images in the time series.
    model_coherence : float
        Model coherence value.
    num_indep_samples : int
        Number of independent samples.

    Returns
    -------
    term1 : np.ndarray
        Precomputed term1 (see above for definition).
    weight_mat : np.ndarray
        Weight matrix (see above for definition).
    """
    coh_mat_stable = np.ones((num_images, num_images), dtype=np.float64) * model_coherence
    coh_mat_stable[np.diag_indices(num_images)] = 1

    inv_coh_mat_stable = np.linalg.inv(coh_mat_stable)
    det_coh_mat_stable = np.linalg.det(coh_mat_stable)

    # precompute term1 and part of term2
    term1 = np.zeros((num_images - 1,), dtype=np.float32)
    weight_mat = np.zeros((num_images - 1, num_images, num_images), dtype=np.float32)

    for step_time_idx in range(1, num_images):
        coh_mat_change = coh_mat_stable.copy()
        coh_mat_change[:step_time_idx, step_time_idx:] = 0
        coh_mat_change[step_time_idx:, :step_time_idx] = 0

        term1[step_time_idx - 1] = num_indep_samples * np.log(
            np.linalg.det(coh_mat_change) / det_coh_mat_stable
        )
        weight_mat[step_time_idx - 1, :, :] = inv_coh_mat_stable - np.linalg.inv(coh_mat_change)
    return term1, weight_mat
