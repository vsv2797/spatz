"""Change detection based on amplitude time series."""
# SpaTZ, SPAtioTemporal Selection of Scatterers
#
# Copyright (c) 2024  Andreas Piter
# (Institute of Photogrammetry and GeoInformation, Leibniz University Hannover)
# Contact: piter@ipi.uni-hannover.de
#
# This software was developed within the context of the PhD thesis of Andreas Piter.
# The software is not yet licensed and used for internal development only.
from scipy import stats
import multiprocessing
import logging
import numpy as np
from matplotlib import pyplot as plt

from mintpy.utils import ptime
import sarvey.utils as ut


def runAmplitude(*,
                 slc: np.ndarray,
                 mask_cand: np.ndarray,
                 significance_level: float,
                 logger: logging.Logger,
                 num_cores: int = 1
                 ):
    """Run change detection based on amplitude time series.

    Steps:
    1. Compute the amplitude time series for each TCS candidate.
    2. Precompute critical values and degrees of freedom for the F-test to increase computation speed.
    3. Start change detection in parallel.

    Parameters
    ----------
    slc: np.ndarray
        SLC stack (number of images x length x width).
    mask_cand: np.ndarray
        Binary mask image indicating the location of TCS candidates that shall be examined (length x width).
    significance_level: float
        Significance level for the F-test.
    logger: logging.Logger
        Logger object.
    num_cores: int
        Number of cores for parallel processing.

    Returns
    -------
    change_time_map: np.ndarray
        Change time map.
    """
    logger.info(msg="AMPLITUDE Change detection.")

    ampl_tcs = np.abs(slc[:, mask_cand])
    num_cand = ampl_tcs.shape[1]

    crit_val_a, crit_val_b, df1, df2 = precomputeCriticalValues(num_samples=slc.shape[0], alpha=significance_level)

    if num_cores == 1:
        parameters = (
            np.arange(num_cand),
            num_cand,
            ampl_tcs,
            crit_val_a,
            crit_val_b,
            df1,
            df2
        )
        idx_range, step_time = launchAmplitude(parameters=parameters)
    else:
        logger.info(msg="start parallel processing with {} cores.".format(num_cores))
        pool = multiprocessing.Pool(processes=num_cores)

        step_time = np.zeros(num_cand, dtype=np.int16)
        num_cores = num_cand if num_cores > num_cand else num_cores  # avoids having more samples than cores
        idx = ut.splitDatasetForParallelProcessing(num_samples=num_cand, num_cores=num_cores)

        args = [(
            idx_range,
            idx_range.shape[0],
            ampl_tcs[:, idx_range],
            crit_val_a,
            crit_val_b,
            df1,
            df2) for idx_range in idx]
        results = pool.map(func=launchAmplitude, iterable=args)

        # retrieve results
        for i, step_time_i in results:
            step_time[i] = step_time_i

    logger.info(f"{num_cand} TCS candidates processed.")

    img_size = slc.shape[1:]
    change_time_map = np.zeros(img_size, dtype=np.int16)
    change_time_map[mask_cand] = step_time
    return change_time_map


def launchAmplitude(parameters: tuple) -> (np.ndarray, np.ndarray):
    """Launch the amplitude change detection for each TCS candidate, intended for use in parallelization.

    Parameters
    ----------
    parameters: tuple
        Tuple containing the following parameters:
        idx_range: np.ndarray
            Index range of the TCS candidates.
        num_cand: int
            Number of TCS candidates.
        ampl_tcs: np.ndarray
            Amplitude time series of the TCS candidates.
        crit_val_a: np.ndarray
            Critical values for the F-test (case a).
        crit_val_b: np.ndarray
            Critical values for the F-test (case b).
        df1: np.ndarray
            Degrees of freedom for the F-test (case a).
        df2: np.ndarray
            Degrees of freedom for the F-test (case b).

    Returns
    -------
    idx_range: np.ndarray
        Index range of the TCS candidates.
    step_time: np.ndarray
        Step time for each TCS candidate.
    """
    # Unpack the parameters
    (idx_range, num_cand, ampl_tcs, crit_val_a, crit_val_b, df1, df2) = parameters

    step_time = np.zeros(num_cand, dtype=np.int16)

    prog_bar = ptime.progressBar(maxValue=num_cand)
    for idx in range(num_cand):
        score = singleCDPrecomputed(
            ts_dat=ampl_tcs[:, idx] * 100,
            crit_val_a=crit_val_a,
            crit_val_b=crit_val_b,
            df1=df1,
            df2=df2
        )
        step_time[idx] = np.nanargmax(score) + 1 if np.nanargmax(score) != 0 else -1
        prog_bar.update(idx + 1, every=np.int16(num_cand / 100),
                        suffix='{}/{} TCS processed.'.format(idx + 1, num_cand))

    return idx_range, step_time


def singleCDPrecomputed(*,
                        ts_dat: np.ndarray,
                        df1: np.ndarray,
                        df2: np.ndarray,
                        crit_val_a: np.ndarray,
                        crit_val_b: np.ndarray) -> np.ndarray:
    """Change detection based on amplitude time series assuming a single change point in the data.

    This method is a precomputed version of the function 'singleCD(...)'.
    Extract most likely step time according to eq. 9 and 10 in Hu et al. (2019).

    Parameters
    ----------
    ts_dat: np.ndarray
        Amplitude time series (number of images x 1).
    df1: np.ndarray
        Degrees of freedom for the F-test (subset 1).
    df2: np.ndarray
        Degrees of freedom for the F-test (subset 2).
    crit_val_a: np.ndarray
        Critical values for the F-test (case a: if test value is > 1).
    crit_val_b: np.ndarray
        Critical values for the F-test (case b: if test value is <= 1).

    Returns
    -------
    score: np.ndarray
        Score for each possible step time. If the F-test is insignificant, the score is set to zero.
    """
    num_samples = ts_dat.shape[0]
    score = np.zeros((num_samples - 1,))

    for p in range(1, num_samples):
        F1 = np.sum(ts_dat[:p] ** 2) / df1[p - 1]
        F2 = np.sum(ts_dat[p:] ** 2) / df2[p - 1]
        if (F1 / F2) > 1:  # case a
            score[p - 1] = F1 / F2 if F1 / F2 > crit_val_a[p - 1] else 0
        else:  # case b
            score[p - 1] = F2 / F1 if F2 / F1 > crit_val_b[p - 1] else 0

    return score


def singleCD(*, ts_dat: np.ndarray, alpha: float, show_plots: bool = False) -> (np.ndarray, int, np.ndarray):
    """Change detection based on amplitude time series assuming a single change point in the data.

    Calling this function multiple times is slow, as the critical values have to be computed for each call. Use the
    precomputed version 'singleCDPrecomputed(...)' instead.
    Extract most likely step time according to eq. 9 and 10 in Hu et al. (2019).

    Parameters
    ----------
    ts_dat: np.ndarray
        Amplitude time series (number of images x 1).
    alpha: float
        Significance level for the F-test.
    show_plots: bool
        Show plots for debugging.

    Returns
    -------
    score: np.ndarray
        Score for each possible step time. If the F-test is insignificant, the score is set to zero.
    change_idx: int
        Index of estimated change point.
    crit_val: np.ndarray
        Critical values for the F-test.
    """
    num_samples = ts_dat.shape[0]
    score = np.zeros((num_samples - 1,))
    crit_val = np.zeros_like(score)

    score_1 = np.zeros((num_samples - 1,))
    crit_val_1 = np.zeros_like(score)

    score_2 = np.zeros((num_samples - 1,))
    crit_val_2 = np.zeros_like(score)

    for p in range(1, num_samples):
        df1 = (2 * p)
        df2 = (2 * (num_samples - p))
        F1 = np.sum(ts_dat[:p] ** 2) / df1
        F2 = np.sum(ts_dat[p:] ** 2) / df2
        if (F1 / F2) > 1:
            score[p - 1] = F1 / F2
            crit_val[p - 1] = stats.f.ppf(1 - alpha, df1, df2)
        else:
            score[p - 1] = F2 / F1
            crit_val[p - 1] = stats.f.ppf(1 - alpha, df2, df1)
        score_1[p - 1] = F1 / F2
        crit_val_1[p - 1] = stats.f.ppf(1 - alpha, df1, df2)
        score_2[p - 1] = F2 / F1
        crit_val_2[p - 1] = stats.f.ppf(1 - alpha, df2, df1)

    if show_plots:
        fig = plt.figure()
        ax1 = fig.add_subplot(4, 1, 1)
        ax2 = fig.add_subplot(4, 1, 2, sharex=ax1)
        ax3 = fig.add_subplot(4, 1, 3, sharex=ax1)
        ax4 = fig.add_subplot(4, 1, 4, sharex=ax1)

        ax1.plot(ts_dat, '.-k', label="Amplitude")
        xsteps = np.linspace(0.5, num_samples - 1.5, num=num_samples - 1)
        ax2.plot(xsteps, score, '.-k', label="F score")
        ylim = ax2.get_ylim()
        ax2.plot(xsteps, crit_val, '-b', label="Critical value")
        ax2.set_ylim(ylim)
        ax2.set_title("Adaptive")

        ax3.plot(xsteps, score_1, label="F score")
        ylim = ax3.get_ylim()
        ax3.plot(xsteps, crit_val_1, label="Critical value")
        ax3.set_ylim(ylim)
        ax3.set_title("F1/F2")

        ax4.plot(xsteps, score_2, label="F score")
        ylim = ax4.get_ylim()
        ax4.plot(xsteps, crit_val_2, label="Critical value")
        ax4.set_ylim(ylim)
        ax4.set_title("F2/F1")

        ax2.set_xlabel("Acquisition no.")
        ax1.set_ylabel("Amplitude")
        ax2.set_ylabel("F score")

    # test if score exceeds critical value
    mask_sign = score > crit_val  # significance
    score[~mask_sign] = 0
    change_idx = np.argmax(score) + 1

    if show_plots:
        ax2.plot(xsteps, score, '.-r', label="significant F score")
        ax2.legend()

    if np.max(score) == 0:
        change_idx = None
        return score, change_idx, crit_val  # no steps found

    if show_plots:
        ax1.plot([np.argmax(score) + 0.5, np.argmax(score) + 0.5], [0, np.max(ts_dat)], ':k')
        ax2.plot([np.argmax(score) + 0.5, np.argmax(score) + 0.5], [0, np.max(score)], ':k')

        fig_hist = plt.figure()
        ax_hist1 = fig_hist.add_subplot(1, 2, 1)
        ax_hist2 = fig_hist.add_subplot(1, 2, 2)
        ax_hist1.hist(ts_dat[:31], bins=20)
        ax_hist2.hist(ts_dat[31:], bins=20)
        plt.show()
    return score, change_idx, crit_val


def multiCD(ts_dat, alpha_distr=0.5, num_bins=10, alpha_step=0.02, min_ts_length=20):
    """Determine step times recursively from times series.

    An empty list is returned if the time series is from the same distribution.
    The implementation of this method is under development and not yet tested.
    Not intended for use in the current version of the software.
    """
    step_list = list()

    if (not isTCSCandidate(alpha=alpha_distr, ts_dat=ts_dat, num_bins=num_bins)) or (ts_dat.shape[0] < min_ts_length):
        return step_list

    score, step_time0, crit_val = singleCD(ts_dat=ts_dat, alpha=alpha_step, show_plots=False)
    if step_time0 is not None:
        step_list.append(step_time0)
        # step_time1 = compute_tcs_step_times(ts_dat[:step_time0], alpha_distr, num_bins, alpha_step, min_ts_length)
        # step_list.extend(step_time0 - step_time1)
        # step_time2 = compute_tcs_step_times(ts_dat[step_time0:], alpha_distr, num_bins, alpha_step, min_ts_length)
        # step_list.extend(step_time0 + step_time2)

    return step_list


def isTCSCandidate(*, alpha: float, ts_dat: np.ndarray, num_bins: int) -> bool:
    """Test if a pixel is a TCS candidate based on the amplitude time series distribution.

    Test if all amplitude observations stem from the same Rayleigh distribution.
    Based on the paper: 'Incorporating Temporary Coherent Scatterers' (Hu et. al. 2019).

    Parameters
    ----------
    alpha: float
        Significance level for the chi-square test.
    ts_dat: np.ndarray
        Amplitude time series (number of images x 1).
    num_bins: int
        Number of bins for the chi-square test.

    Returns
    -------
    bool
        True if the null hypothesis is rejected, False otherwise
    """
    num_samples = ts_dat.shape[0]
    sigma_sqr = 1 / (2 * num_samples) * np.sum(ts_dat ** 2)
    bins = [np.sqrt(2 * sigma_sqr * np.log(num_bins / (num_bins - i))) for i in range(num_bins)]
    # lower bound is already included as the loop starts from zero, but last upper bound needs to be added
    bins.append(np.inf)

    bin_count, _ = np.histogram(ts_dat, bins)
    bin_count = np.array(bin_count)
    test_val = np.sum((bin_count - num_samples / num_bins) ** 2 / (num_samples / num_bins))
    df = num_bins - 2
    crit_val = stats.chi2.ppf(1 - alpha, df)
    if test_val > crit_val:
        return True
    else:
        return False


def precomputeCriticalValues(*, num_samples: int, alpha: float) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """Precompute critical values and degrees of freedom for the F-test.

    The critical values are precomputed for all possible step times to increase computation speed.

    Parameters
    ----------
    num_samples: int
        Number of samples in the time series.
    alpha: float
        Significance level for the F-test.

    Returns
    -------
    crit_val_a: np.ndarray
        Critical values for the F-test (case a: if test value is > 1).
    crit_val_b: np.ndarray
        Critical values for the F-test (case b: if test value is <= 1).
    df1: np.ndarray
        Degrees of freedom for the F-test (subset 1).
    df2: np.ndarray
        Degrees of freedom for the F-test (subset 2).
    """
    crit_val_a = np.zeros((num_samples - 1,))
    crit_val_b = np.zeros((num_samples - 1,))
    df1 = np.zeros((num_samples - 1,))
    df2 = np.zeros((num_samples - 1,))

    for p in range(1, num_samples):
        df1[p - 1] = (2 * p)
        df2[p - 1] = (2 * (num_samples - p))
        crit_val_a[p - 1] = stats.f.ppf(1 - alpha, df1[p - 1], df2[p - 1])
        crit_val_b[p - 1] = stats.f.ppf(1 - alpha, df2[p - 1], df1[p - 1])

    return crit_val_a, crit_val_b, df1, df2
