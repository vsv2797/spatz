"""Change detection with spatial phase noise estimation from the interferometric phases."""
# SpaTZ, SPAtioTemporal Selection of Scatterers
#
# Copyright (c) 2024  Andreas Piter
# (Institute of Photogrammetry and GeoInformation, Leibniz University Hannover)
# Contact: piter@ipi.uni-hannover.de
#
# This software was developed within the context of the PhD thesis of Andreas Piter.
# The software is not yet licensed and used for internal development only.
import multiprocessing
import logging
import numpy as np
from matplotlib import pyplot as plt

from miaplpy.objects.slcStack import slcStack
from mintpy.utils import ptime

from sarvey.coherence import computeIfgs
from sarvey.ifg_network import SmallTemporalBaselinesNetwork, IfgNetwork
import sarvey.utils as ut

from .likelihood import logLikelihoodInterferometricPhase, marginalPDFInterferometricPhase


def runSpatialNoiseEstimation(*,
                              slc_stack_obj: slcStack,
                              slc: np.ndarray,
                              mask_cand: np.ndarray,
                              logger: logging.Logger,
                              wdw_size: int = 9,
                              num_cores: int = 1,
                              num_link: int = 5
                              ) -> (np.ndarray, np.ndarray, np.ndarray):
    """Run change detection based on phase noise estimation with a spatial displacement model.

    Steps:
    1. Create a small temporal baseline network.
    2. Precompute the interferogram indices for each possible temporal subset.
    3. Launch the spatial noise estimation for each TCS candidate in parallel.
    4. Retrieve the results.

    Parameters
    ----------
    slc_stack_obj : slcStack
        SLC stack object.
    slc : np.ndarray
        SLC stack (number SLC images x length x width).
    mask_cand : np.ndarray
        Binary mask image indicating the location of TCS candidates that shall be examined (length x width).
    logger : logging.Logger
        Logger object for logging information.
    wdw_size : int, optional
        Size of the window for spatial lowpass filtering (averaging) (default is 9 pixels).
    num_cores : int, optional
        Number of CPU cores to use for parallel processing (default is 1).
    num_link : int, optional
        Number of links (interferograms) for the interferogram network (default is 5).

    Returns
    -------
    change_time_map : np.ndarray
        Image with the estimated change time index for each pixel (length x width). Change time is within interval
        [0, number of images - 1]. The change time index is np.nan for pixels that could not be processed.
    coherence_map : np.ndarray
        Image with the coherence for each pixel (length x width) after change detection. The coherence is evaluated for
        each of the two temporal subsets resulting from the change time. The returned coherence belongs to the maximum
        coherence of the two subsets.
    coherence_initial_map : np.ndarray
        Image with the coherence for each pixel (length x width) before change detection. The coherence is evaluated for
        the whole time span of the SLC stack.
    """
    logger.info(msg="SPATIAL NOISE ESTIMATION Change detection.")

    wdw_hfs = int(wdw_size // 2)
    num_cand = (np.where(mask_cand)[0]).shape[0]
    coord_tcs_cand = np.where(mask_cand)
    coord_x_cand = coord_tcs_cand[0]
    coord_y_cand = coord_tcs_cand[1]
    step_time = np.zeros(num_cand, dtype=np.int16)
    max_coherence = np.zeros(num_cand, dtype=np.float32)
    initial_coherence = np.zeros(num_cand, dtype=np.float32)

    cent_idx = int(wdw_size / 2)
    siblings = np.ones((wdw_size, wdw_size), dtype=np.bool_)
    siblings[cent_idx, cent_idx] = False

    # logger.info(msg="Create small temporal baseline + yearly ifgs network")
    # ifg_net_obj = SmallBaselineYearlyNetwork()
    # ifg_net_obj.configure(
    #     pbase=slc_stack_obj.pbase,
    #     tbase=slc_stack_obj.tbase,
    #     num_link=num_link,
    #     dates=slc_stack_obj.dateList
    # )

    logger.info(msg="Create small temporal baseline network")
    ifg_net_obj = SmallTemporalBaselinesNetwork()
    ifg_net_obj.configure(
        pbase=slc_stack_obj.pbase,
        tbase=slc_stack_obj.tbase,
        num_link=num_link,
        dates=slc_stack_obj.dateList
    )

    global global_slc
    global_slc = slc

    related_ifgs1, related_ifgs2 = precomputeIfgIndices(ifg_net_obj=ifg_net_obj)

    if num_cores == 1:
        parameters = (
            np.arange(num_cand),
            num_cand,
            coord_x_cand,
            coord_y_cand,
            wdw_hfs,
            ifg_net_obj,
            siblings,
            related_ifgs1,
            related_ifgs2
        )
        idx_range, step_time, max_coherence, initial_coherence = launchSpatialNoiseEstimation(
            parameters=parameters)
    else:
        logger.info(msg="start parallel processing with {} cores.".format(num_cores))
        pool = multiprocessing.Pool(processes=num_cores)
        num_cores = num_cand if num_cores > num_cand else num_cores  # avoids having more samples than cores
        idx = ut.splitDatasetForParallelProcessing(num_samples=num_cand, num_cores=num_cores)
        args = [(
            idx_range,
            idx_range.shape[0],
            coord_x_cand[idx_range],
            coord_y_cand[idx_range],
            wdw_hfs,
            ifg_net_obj,
            siblings,
            related_ifgs1,
            related_ifgs2
        ) for idx_range in idx]

        results = pool.map(func=launchSpatialNoiseEstimation, iterable=args)

        # retrieve results
        for i, step_time_i, max_coh_i, coh_0_i in results:
            step_time[i] = step_time_i
            max_coherence[i] = max_coh_i
            initial_coherence[i] = coh_0_i

    logger.info(f"{num_cand} TCS candidates processed.")

    img_size = slc.shape[1:]
    change_time_map = np.zeros(img_size, dtype=np.int16)
    change_time_map[mask_cand] = step_time

    coherence_map = np.zeros(img_size, dtype=np.float32)
    coherence_map[mask_cand] = max_coherence

    coherence_initial_map = np.zeros(img_size, dtype=np.float32)
    coherence_initial_map[mask_cand] = initial_coherence
    return change_time_map, coherence_map, coherence_initial_map


def launchSpatialNoiseEstimation(parameters: tuple) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """Launch the spatial noise estimation for each TCS candidate, intended for use in parallelization.

    Steps:
    For each TCS candidate:
    1. Extract the SLC window around the TCS candidate.
    2. Compute the interferograms for the SLC window.
    3. Spatially lowpass filter the interferograms to estimate the phase noise.
    4. Apply change detection with assumption of a single change in the time series and estimate the step time.

    Parameters
    ----------
    parameters : tuple
        Tuple with the following elements:
        idx_range : np.ndarray
            Indices of the TCS candidates that shall be processed.
        num_cand : int
            Number of TCS candidates.
        coord_x_cand : np.ndarray
            x-coordinates of the TCS candidates.
        coord_y_cand : np.ndarray
            y-coordinates of the TCS candidates.
        wdw_hfs : int
            Half window size for spatial lowpass filtering.
        ifg_net_obj : IfgNetwork
            Interferogram network object.
        siblings : np.ndarray
            Boolean array indicating the siblings of the central pixel.
        related_ifgs1 : dict
            Dictionary with the interferogram indices for the first temporal subset.
        related_ifgs2 : dict
            Dictionary with the interferogram indices for the second temporal subset.

    Returns
    -------
    idx_range : np.ndarray
        Indices of the TCS candidates that have been processed.
    step_time : np.ndarray
        Estimated step time for each TCS candidate (number of TCS x 1).
    max_coherence : np.ndarray
        Coherence at the estimated change time (number of TCS x 1). The temporal subset with the higher coherence is
        chosen.
    initial_coherence : np.ndarray
        Coherence for the whole time span (number of TCS x 1).
    """
    (idx_range, num_cand, coord_x_cand, coord_y_cand, wdw_hfs, ifg_net_obj, siblings, related_ifgs1,
     related_ifgs2) = parameters

    step_time = np.zeros(num_cand, dtype=np.int16)
    max_coherence = np.zeros(num_cand, dtype=np.float32)
    initial_coherence = np.zeros(num_cand, dtype=np.float32)

    prog_bar = ptime.progressBar(maxValue=num_cand)
    for idx in range(num_cand):
        x = coord_x_cand[idx]
        y = coord_y_cand[idx]

        y1 = y - wdw_hfs if y - wdw_hfs >= 0 else 0
        y2 = y + wdw_hfs + 1 if y + wdw_hfs + 1 <= global_slc.shape[2] else global_slc.shape[2]
        x1 = x - wdw_hfs if x - wdw_hfs >= 0 else 0
        x2 = x + wdw_hfs + 1 if x + wdw_hfs + 1 <= global_slc.shape[1] else global_slc.shape[1]
        slc_wdw = global_slc[:, x1:x2, y1:y2]

        if (slc_wdw.shape[1] != (wdw_hfs * 2 + 1)) | (slc_wdw.shape[2] != (wdw_hfs * 2 + 1)):
            # skip problems at the boundary of the image.
            step_time[idx] = -1
            prog_bar.update(idx + 1, every=np.int16(num_cand / 200),
                            suffix='{}/{} candidates processed.'.format(idx + 1, num_cand))
            continue

        ifgs = computeIfgs(slc=slc_wdw, ifg_array=np.array(ifg_net_obj.ifg_list))

        # filter ifgs and estimate noise
        avg_neighbours = np.mean(ifgs[siblings, :], axis=0)

        ifg_lowpass = ifgs[~siblings, :] * np.conjugate(avg_neighbours)
        ifg_res = np.angle(ifg_lowpass).ravel()

        score, max_coh, coherence_0 = singleCDPrecomputed(
            ifg_res=ifg_res,
            ifg_net_obj=ifg_net_obj,
            related_ifgs1=related_ifgs1,
            related_ifgs2=related_ifgs2
        )

        step_time[idx] = np.nanargmin(score) + 1
        max_coherence[idx] = max_coh
        initial_coherence[idx] = coherence_0

        prog_bar.update(idx + 1, every=np.int16(num_cand / 200),
                        suffix='{}/{} candidates processed.'.format(idx + 1, num_cand))
    prog_bar.close()
    return idx_range, step_time, max_coherence, initial_coherence


def singleCDPrecomputed(*, ifg_res: np.ndarray, ifg_net_obj: IfgNetwork,
                        related_ifgs1: dict, related_ifgs2: dict) -> (np.ndarray, float, float):
    """Change detection with spatial phase noise estimation assuming a single change point in the data.

    This method is a precomputed version of the function 'singleCD(...)'.

    Steps:
    1. Compute the likelihood ratio test for each possible change time.
    2. Normalize the score by the number of interferograms in each temporal subset.
    3. Retrieve the maximum coherence at the estimated change time.

    Parameters
    ----------
    ifg_res : np.ndarray
        Phase residuals for the interferograms (number of interferograms x 1).
    ifg_net_obj : IfgNetwork
        Interferogram network object.
    related_ifgs1 : dict
        Dictionary with the interferogram indices for the first temporal subset.
    related_ifgs2 : dict
        Dictionary with the interferogram indices for the second temporal subset.

    Returns
    -------
    score_norm_ifgs : np.ndarray
        Normalized score for each possible change time (number of images - 1 x 1).
    max_coh : float
        Coherence from the temporal subset with higher coherence at estimated change time.
    coherence_0 : float
        Coherence for the whole time span.
    """
    score_norm_ifgs = np.zeros((ifg_net_obj.num_images - 1,))
    score_norm_ifgs[:] = np.nan  # indicate values which are not computed by nan

    phase_0 = 0  # expected value of phase is assumed to be Zero
    num_0 = ifg_net_obj.num_ifgs
    coherence_0 = np.abs(np.mean(np.exp(1j * ifg_res)))

    p0 = logLikelihoodInterferometricPhase(coherence=coherence_0, ifg_res=ifg_res, phase_0=phase_0)
    """
    The LRT is not computed for the first and last possible subsets which have only two images each. Hence, there is
    only one interferogram possible which yields a coherence of 1 and the marginal distribution is 0. As log(0) is -inf,
    the score is not valid for these change times. As a consequence, the minimum size of a subset is 3 images.
    """
    for step_time_idx in range(2, ifg_net_obj.num_images - 3):
        phase_noise_1 = ifg_res[related_ifgs1[step_time_idx]]
        coherence_1 = np.abs(np.mean(np.exp(1j * phase_noise_1)))

        phase_noise_2 = ifg_res[related_ifgs2[step_time_idx]]
        coherence_2 = np.abs(np.mean(np.exp(1j * phase_noise_2)))

        num_1 = phase_noise_1.shape[0]
        num_2 = phase_noise_2.shape[0]
        if (num_1 == 1) or (num_2 == 1):
            print("WARNING: Subset has only 1 interferogram.")
        p1 = logLikelihoodInterferometricPhase(coherence=coherence_1, ifg_res=phase_noise_1, phase_0=phase_0)
        p2 = logLikelihoodInterferometricPhase(coherence=coherence_2, ifg_res=phase_noise_2, phase_0=phase_0)

        score_norm_ifgs[step_time_idx] = (p0 / num_0 - ((p1 + p2) / (num_1 + num_2)))

    phase_noise_1 = ifg_res[related_ifgs1[np.nanargmin(score_norm_ifgs)]]
    phase_noise_2 = ifg_res[related_ifgs2[np.nanargmin(score_norm_ifgs)]]
    max_coh = max(np.abs(np.mean(np.exp(1j * phase_noise_1))), np.abs(np.mean(np.exp(1j * phase_noise_2))))
    return score_norm_ifgs, max_coh, coherence_0


def singleCD(*, ifg_res: np.ndarray, ifg_net_obj: IfgNetwork, show_plots: bool = False) -> np.ndarray:
    """Change detection with spatial phase noise estimation assuming a single change point in the data.

    Calling this function multiple times is slow, as the interferogram indices are computed for each call. Use the
    precomputed version 'singleCDPrecomputed(...)' instead.

    Steps:
    1. Compute the likelihood ratio test for each possible change time.
    2. Normalize the score by the number of interferograms in each temporal subset.

    Parameters
    ----------
    ifg_res : np.ndarray
        Phase residuals for the interferograms (number of interferograms x 1).
    ifg_net_obj : IfgNetwork
        Interferogram network object.
    show_plots : bool, optional
        Flag to show plots for debugging (default is False).

    Returns
    -------
    score_norm_ifgs : np.ndarray
        Normalized score for each possible change time (number of images - 1 x 1).
    """
    design_mat = ifg_net_obj.getDesignMatrix()
    score_norm_ifgs = np.zeros((ifg_net_obj.num_images - 1,))
    score_norm_ifgs[:] = np.nan  # indicate values which are not computed by nan
    score_norm_imgs = score_norm_ifgs.copy()
    # score = score_norm_ifgs.copy()
    coherences = np.zeros((ifg_net_obj.num_images - 1, 2))
    loglikelihoods = np.zeros((ifg_net_obj.num_images - 1, 2))
    # ps = np.zeros((ifg_net_obj.num_images - 1, 2))

    phase_0 = 0  # expected value of phase is assumed to be Zero
    num_0 = ifg_net_obj.num_ifgs
    coherence_0 = np.abs(np.mean(np.exp(1j * ifg_res)))

    p0 = logLikelihoodInterferometricPhase(coherence=coherence_0, ifg_res=ifg_res, phase_0=phase_0)

    num_ifgs_per_subset = np.zeros((ifg_net_obj.num_images - 1, 2))
    num_imgs_per_subset = np.zeros((ifg_net_obj.num_images - 1, 2))

    if show_plots:
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        xval = np.linspace(-np.pi, np.pi, 100)
        yval = marginalPDFInterferometricPhase(coherence=coherence_0, phase=xval, phase_0=phase_0)
        ax1.plot(xval, yval)
        ax2.hist(ifg_res, bins=100, density=True)
    """
    The LRT is not computed for the first and last possible subsets which have only two images each. Hence, there is
    only one interferogram possible which yields a coherence of 1 and the marginal distribution is 0. As log(0) is -inf,
    the score is not valid for these change times. As a consequence, the minimum size of a subset is 3 images.
    """
    for step_time_idx in range(2, ifg_net_obj.num_images - 3):
        phase_noise_1, coherence_1 = getPhaseResidualsForTimeSpan(
            ifg_res=ifg_res,
            design_mat=design_mat,
            ifg_net_obj=ifg_net_obj,
            start_idx=0,
            stop_idx=step_time_idx
        )
        phase_noise_2, coherence_2 = getPhaseResidualsForTimeSpan(
            ifg_res=ifg_res,
            design_mat=design_mat,
            ifg_net_obj=ifg_net_obj,
            start_idx=step_time_idx + 1,
            stop_idx=ifg_net_obj.num_images - 1
        )
        num_1 = phase_noise_1.shape[0]
        num_2 = phase_noise_2.shape[0]
        num_images_1 = step_time_idx + 1
        num_images_2 = ifg_net_obj.num_images - num_images_1
        if (num_1 == 1) or (num_2 == 1):
            print("WARNING: Subset has only 1 interferogram.")
        num_ifgs_per_subset[step_time_idx, 0] = num_1
        num_ifgs_per_subset[step_time_idx, 1] = num_2
        num_imgs_per_subset[step_time_idx, 0] = num_images_1
        num_imgs_per_subset[step_time_idx, 1] = num_images_2
        coherences[step_time_idx, 0] = coherence_1
        coherences[step_time_idx, 1] = coherence_2
        p1 = logLikelihoodInterferometricPhase(coherence=coherence_1, ifg_res=phase_noise_1, phase_0=phase_0)
        p2 = logLikelihoodInterferometricPhase(coherence=coherence_2, ifg_res=phase_noise_2, phase_0=phase_0)
        # p1 = p1 if p1 < 1 else 1
        # p2 = p2 if p2 < 1 else 1

        # ps[step_time_idx, 0] = p1
        # ps[step_time_idx, 1] = p2

        loglikelihoods[step_time_idx, 0] = p1
        loglikelihoods[step_time_idx, 1] = p2

        # score[step_time_idx] = p0 - p1 + p2  # not normalized
        score_norm_ifgs[step_time_idx] = (p0 / num_0 -
                                          ((p1 + p2) / (num_1 + num_2)))
        score_norm_imgs[step_time_idx] = (p0 / ifg_net_obj.num_images -
                                          ((p1 + p2) / (num_images_1 + num_images_2)))  # normalized

    if show_plots:
        fig = plt.figure()
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)
        ax1.plot(num_ifgs_per_subset[:, 0], label='num_ifgs_1')
        ax1.plot(num_ifgs_per_subset[:, 1], label='num_ifgs_2')
        ax1.legend()

        ax2.plot(num_imgs_per_subset[:, 0], label='num_img_1')
        ax2.plot(num_imgs_per_subset[:, 1], label='num_img_2')
        ax2.legend()

        ax3.plot(coherences[:, 0], label='coherence_1')
        ax3.plot(coherences[:, 1], label='coherence_2')
        ax3.legend()

        ax4.plot(loglikelihoods[:, 0], label='loglikelihood_1')
        ax4.plot(loglikelihoods[:, 1], label='loglikelihood_2')
        ax4.legend()

        # ax5.plot(ps[:, 0], label='pdf_1')
        # ax5.plot(ps[:, 1], label='pdf_2')
        # ax5.legend()

        ratio = coherences[:, 0] / coherences[:, 1]
        plt.figure()
        plt.plot(ratio)

        plt.figure()
        # plt.plot(score, label='score - without normalization')
        plt.plot(score_norm_ifgs, label='score - normalized by #ifgs')
        plt.plot(score_norm_imgs, label='score - normalized by #images')
        plt.legend()
        plt.show()

    # return np.nanargmin(score) + 1
    return score_norm_ifgs


def getIfgIndicesForTimeSpan(*,
                             design_mat: np.ndarray,
                             ifg_net_obj: IfgNetwork,
                             start_idx: int,
                             stop_idx: int) -> np.ndarray:
    """Retrieve the indices of all interferograms belonging to the given time span.

    Parameters
    ----------
    design_mat : np.ndarray
        Design matrix of the interferogram network (number of interferograms x number of images).
    ifg_net_obj : IfgNetwork
        Interferogram network object.
    start_idx : int
        Start index of the time span.
    stop_idx : int
        Stop index of the time span.

    Returns
    -------
    related_ifgs : np.ndarray
        Indices of the interferograms that belong to the given time span.
    """
    cand_ifgs = [np.where(design_mat[:, j])[0].flatten() for j in range(start_idx, stop_idx)]
    if len(cand_ifgs) != 0:
        cand_ifgs = np.unique(np.hstack(cand_ifgs))
    else:
        related_ifgs = np.empty(0)
        return related_ifgs

    # find ifgs which are within this time span (not the ones connecting to images outside this time)
    start_time = ifg_net_obj.tbase[start_idx]
    stop_time = ifg_net_obj.tbase[stop_idx]
    related_ifgs = list()
    for ifg_idx in cand_ifgs:
        image_0 = np.where(design_mat[ifg_idx, :])[0][0]
        image_1 = np.where(design_mat[ifg_idx, :])[0][1]
        start_inside = ((ifg_net_obj.tbase[image_0] >= start_time) &
                        (ifg_net_obj.tbase[image_0] <= stop_time))
        stop_inside = ((ifg_net_obj.tbase[image_1] >= start_time) &
                       (ifg_net_obj.tbase[image_1] <= stop_time))
        if start_inside & stop_inside:
            related_ifgs.append(ifg_idx)
    related_ifgs = np.array(related_ifgs)

    return related_ifgs


def precomputeIfgIndices(*, ifg_net_obj: IfgNetwork) -> (dict, dict):
    """Precompute the interferogram indices for each possible temporal subset.

    Parameters
    ----------
    ifg_net_obj : IfgNetwork
        Interferogram network object.

    Returns
    -------
    related_ifgs1 : dict
        Dictionary with the interferogram indices for the first temporal subset.
    related_ifgs2 : dict
        Dictionary with the interferogram indices for the second temporal subset.
    """
    design_mat = ifg_net_obj.getDesignMatrix()

    related_ifgs1 = dict()
    related_ifgs2 = dict()

    for step_time_idx in range(2, ifg_net_obj.num_images - 3):
        related_ifgs1[step_time_idx] = getIfgIndicesForTimeSpan(
            design_mat=design_mat,
            ifg_net_obj=ifg_net_obj,
            start_idx=0,
            stop_idx=step_time_idx
        )
        related_ifgs2[step_time_idx] = getIfgIndicesForTimeSpan(
            design_mat=design_mat,
            ifg_net_obj=ifg_net_obj,
            start_idx=step_time_idx + 1,
            stop_idx=ifg_net_obj.num_images - 1
        )

    return related_ifgs1, related_ifgs2


def getPhaseResidualsForTimeSpan(*,
                                 ifg_res: np.ndarray,
                                 design_mat: np.ndarray,
                                 ifg_net_obj: IfgNetwork,
                                 start_idx: int,
                                 stop_idx: int) -> (np.ndarray, float):
    """Retrieve the phase residuals for a given time span and compute temporal coherence.

    This function is very similar to 'getIfgIndicesForTimeSpan(...)', but it returns the phase residuals and the
    temporal coherence for use in 'singleCD(...)', while 'getIfgIndicesForTimeSpan(...)' is intended for accelerating
    the computations in 'singleCDPrecomputed(...)'.

    Parameters
    ----------
    ifg_res : np.ndarray
        Phase residuals for the interferograms (number of interferograms x 1).
    design_mat : np.ndarray
        Design matrix of the interferogram network (number of interferograms x number of images).
    ifg_net_obj : IfgNetwork
        Interferogram network object.
    start_idx : int
        Start index of the time span.
    stop_idx : int
        Stop index of the time span.

    Returns
    -------
    ifg_res : np.ndarray
        Phase residuals for the interferograms that belong to the given time span.
    temp_coh : float
        Temporal coherence for the given time span.
    """
    cand_ifgs = [np.where(design_mat[:, j])[0].flatten() for j in range(start_idx, stop_idx)]
    if len(cand_ifgs) != 0:
        cand_ifgs = np.unique(np.hstack(cand_ifgs))
    else:
        temp_coh = np.nan
        ifg_res = np.empty(0)
        return ifg_res, temp_coh

    # find ifgs which are within this time span (not the ones connecting to images outside this time)
    start_time = ifg_net_obj.tbase[start_idx]
    stop_time = ifg_net_obj.tbase[stop_idx]
    related_ifgs = list()
    for ifg_idx in cand_ifgs:
        image_0 = np.where(design_mat[ifg_idx, :])[0][0]
        image_1 = np.where(design_mat[ifg_idx, :])[0][1]
        start_inside = ((ifg_net_obj.tbase[image_0] >= start_time) &
                        (ifg_net_obj.tbase[image_0] <= stop_time))
        stop_inside = ((ifg_net_obj.tbase[image_1] >= start_time) &
                       (ifg_net_obj.tbase[image_1] <= stop_time))
        if start_inside & stop_inside:
            related_ifgs.append(ifg_idx)
    related_ifgs = np.array(related_ifgs)
    temp_coh = np.abs(np.mean(np.exp(1j * ifg_res[related_ifgs])))
    ifg_res = ifg_res[related_ifgs]
    return ifg_res, temp_coh
