"""Coherent lifetime analysis."""
# SpaTZ, SPAtioTemporal Selection of Scatterers
#
# Copyright (c) 2024  Andreas Piter
# (Institute of Photogrammetry and GeoInformation, Leibniz University Hannover)
# Contact: piter@ipi.uni-hannover.de
#
# This software was developed within the context of the PhD thesis of Andreas Piter.
# The software is not yet licensed and used for internal development only.
import logging
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
import cmcrameri as cmc

from miaplpy.objects import slcStack
from mintpy.utils import ptime
from mintpy.utils.plot import auto_flip_direction

from sarvey.coherence import computeIfgs
from sarvey.ifg_network import SmallTemporalBaselinesNetwork
import sarvey.utils as ut

from ..change.phase_noise_space import precomputeIfgIndices
from ..utils.misc import FigureSaveHandler


def runLifetimeEstimation(*,
                          slc_stack_obj: slcStack,
                          slc: np.ndarray,
                          mask_cand: np.ndarray,
                          change_time_map: np.ndarray,
                          logger: logging.Logger,
                          wdw_size: int = 9,
                          num_link: int = 5,
                          num_cores: int = 1,
                          ):
    """LifetimeEstimation.

    This function is very similar to change.phase_noise.runSpatialNoiseEstimation.
    """
    wdw_hfs = int(wdw_size // 2)
    num_cand = (np.where(mask_cand)[0]).shape[0]
    coord_tcs_cand = np.where(mask_cand)
    coord_x_cand = coord_tcs_cand[0]
    coord_y_cand = coord_tcs_cand[1]
    max_coherence = np.zeros(num_cand, dtype=np.float64)
    initial_coherence = np.zeros(num_cand, dtype=np.float64)
    subset_length = np.zeros(num_cand, dtype=np.int64)
    subset_index = np.zeros(num_cand, dtype=np.int32)

    change_time = change_time_map[mask_cand]

    cent_idx = int(wdw_size / 2)
    siblings = np.ones((wdw_size, wdw_size), dtype=np.bool_)
    siblings[cent_idx, cent_idx] = False

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
            related_ifgs2,
            change_time
        )
        idx_range, max_coherence, initial_coherence, subset_length, subset_index = launchLifetimeEstimation(
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
            related_ifgs2,
            change_time[idx_range]
        ) for idx_range in idx]

        results = pool.map(func=launchLifetimeEstimation, iterable=args)

        # retrieve results
        for i, max_coh_i, coh_0_i, sub_len_i, sub_idx_i in results:
            max_coherence[i] = max_coh_i
            initial_coherence[i] = coh_0_i
            subset_length[i] = sub_len_i
            subset_index[i] = sub_idx_i

    logger.info(f"{num_cand} TCS candidates processed.")

    img_size = slc.shape[1:]
    coherence_map = np.zeros(img_size, dtype=np.float64)
    coherence_map[mask_cand] = max_coherence

    coherence_initial_map = np.zeros(img_size, dtype=np.float64)
    coherence_initial_map[mask_cand] = initial_coherence

    subset_length_map = np.zeros(img_size, dtype=np.int64)
    subset_length_map[mask_cand] = subset_length

    subset_index_map = np.zeros(img_size, dtype=np.int32)
    subset_index_map[mask_cand] = subset_index

    return coherence_map, coherence_initial_map, subset_length_map, subset_index_map


def launchLifetimeEstimation(parameters: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """LaunchLifetimeEstimation."""
    (idx_range, num_cand, coord_x_cand, coord_y_cand, wdw_hfs, ifg_net_obj, siblings, related_ifgs1,
     related_ifgs2, change_time) = parameters

    max_coherence = np.zeros(num_cand, dtype=np.float64)
    initial_coherence = np.zeros(num_cand, dtype=np.float64)
    subset_length = np.zeros(num_cand, dtype=np.int64)
    subset_index = np.zeros(num_cand, dtype=np.int32)

    prog_bar = ptime.progressBar(maxValue=num_cand)
    for idx in range(num_cand):
        if change_time[idx] == -1:
            prog_bar.update(idx + 1, every=np.int16(num_cand / 200),
                            suffix='{}/{} candidates processed.'.format(idx + 1, num_cand))
            continue

        x = coord_x_cand[idx]
        y = coord_y_cand[idx]

        y1 = y - wdw_hfs if y - wdw_hfs >= 0 else 0
        y2 = y + wdw_hfs + 1 if y + wdw_hfs + 1 <= global_slc.shape[2] else global_slc.shape[2]
        x1 = x - wdw_hfs if x - wdw_hfs >= 0 else 0
        x2 = x + wdw_hfs + 1 if x + wdw_hfs + 1 <= global_slc.shape[1] else global_slc.shape[1]
        slc_wdw = global_slc[:, x1:x2, y1:y2]

        if (slc_wdw.shape[1] != (wdw_hfs * 2 + 1)) | (slc_wdw.shape[2] != (wdw_hfs * 2 + 1)):
            # skip problems at the boundary of the image.
            prog_bar.update(idx + 1, every=np.int16(num_cand / 200),
                            suffix='{}/{} candidates processed.'.format(idx + 1, num_cand))
            continue

        ifgs = computeIfgs(slc=slc_wdw, ifg_array=np.array(ifg_net_obj.ifg_list))
        avg_neighbours = np.mean(ifgs[siblings, :], axis=0)
        ifg_lowpass = ifgs[~siblings, :] * np.conjugate(avg_neighbours)
        ifg_res = np.angle(ifg_lowpass).ravel()

        try:
            phase_noise_1 = ifg_res[related_ifgs1[change_time[idx]]]
            coherence_1 = np.abs(np.mean(np.exp(1j * phase_noise_1)))

            phase_noise_2 = ifg_res[related_ifgs2[change_time[idx]]]
            coherence_2 = np.abs(np.mean(np.exp(1j * phase_noise_2)))
            if coherence_2 > 1 or coherence_1 > 1:
                print("Error: coherence > 1")
                print("coherence_1: ", coherence_1)
                print("coherence_2: ", coherence_2)
                print("phase_noise_1: ", phase_noise_1)
                print("phase_noise_2: ", phase_noise_2)
                print("change_time:", change_time[idx])

        except KeyError as kerr:
            print(f"Detected change point is either at first or last two images and coherence cannot be computed:"
                  f"\n{kerr}")
            prog_bar.update(idx + 1, every=np.int16(num_cand / 200),
                            suffix='{}/{} candidates processed.'.format(idx + 1, num_cand))
            continue

        length = [change_time[idx], ifg_net_obj.num_images - change_time[idx]]  # length of each subset

        initial_coherence[idx] = np.abs(np.mean(np.exp(1j * ifg_res)))
        max_coherence[idx] = np.max([coherence_1, coherence_2])
        subset_index[idx] = np.argmax([coherence_1, coherence_2])
        subset_length[idx] = length[subset_index[idx]]
        prog_bar.update(idx + 1, every=np.int16(num_cand / 200),
                        suffix='{}/{} candidates processed.'.format(idx + 1, num_cand))
    prog_bar.close()
    return idx_range, max_coherence, initial_coherence, subset_length, subset_index


def analyseLifetime(*,
                    coherence_initial_map: np.ndarray,
                    coherence_map: np.ndarray,
                    slc_stack_obj: slcStack,
                    mask_cand: np.ndarray,
                    change_time_map: np.ndarray,
                    subset_length_map: np.ndarray,
                    subset_index_map: np.ndarray,
                    logger: logging.Logger,
                    fig_save_obj: FigureSaveHandler):
    """AnalyseLifetime."""
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
    img0 = axs[0].imshow(coherence_initial_map, vmin=0, vmax=1, cmap=cmc.cm.cmaps["grayC"], interpolation="nearest")
    axs[0].set_title("Initial coherence")
    img1 = axs[1].imshow(coherence_map, vmin=0, vmax=1, cmap=cmc.cm.cmaps["grayC"], interpolation="nearest")
    axs[1].set_title("Coherence of subset")
    coh_diff_map = coherence_map - coherence_initial_map
    img2 = axs[2].imshow(coh_diff_map,
                         vmin=-max(abs(coh_diff_map.ravel())),
                         vmax=max(abs(coh_diff_map.ravel())),
                         cmap=cmc.cm.cmaps["vik_r"],
                         interpolation="nearest")
    axs[2].set_title("Difference in coherence (subset - initial)")
    plt.colorbar(img0, ax=axs[0], fraction=0.02, pad=0.04)
    plt.colorbar(img1, ax=axs[1], fraction=0.02, pad=0.04)
    plt.colorbar(img2, ax=axs[2], fraction=0.02, pad=0.04)
    auto_flip_direction(slc_stack_obj.metadata, ax=axs[0], print_msg=False)
    auto_flip_direction(slc_stack_obj.metadata, ax=axs[1], print_msg=False)
    auto_flip_direction(slc_stack_obj.metadata, ax=axs[2], print_msg=False)
    fig_save_obj.saveAndClose("0_coherence_map")

    fig, axs = plt.subplots(1, 3)
    mask_valid = mask_cand & (change_time_map > 0)
    axs[0].hist(coherence_map[mask_valid], bins=100)
    axs[0].set_xlabel("Coherence of subset")
    axs[0].set_ylabel("Frequency")
    axs[1].hist(change_time_map[mask_valid], bins=100)
    axs[1].set_xlabel("Change time index")
    axs[1].set_ylabel("Frequency")
    sca = axs[2].scatter(change_time_map[mask_valid], coherence_map[mask_valid], s=2, c=coh_diff_map[mask_valid],
                         cmap=cmc.cm.cmaps["vik_r"], vmin=-max(abs(coh_diff_map.ravel())),
                         vmax=max(abs(coh_diff_map.ravel())))
    cbar = plt.colorbar(sca, ax=axs[2], fraction=0.02, pad=0.04)
    cbar.set_label("Difference in coherence\n(subset - initial)")
    axs[2].set_xlabel("Change time index")
    axs[2].set_ylabel("Coherence of subset")
    fig_save_obj.saveAndClose("1_histograms")

    fig, axs = plt.subplots(1, 2)
    sca = axs[0].scatter(coherence_initial_map[mask_cand], coherence_map[mask_cand], s=1.5,
                         c=change_time_map[mask_cand], cmap=cmc.cm.cmaps["batlow"])
    axs[0].set_xlabel("Initial coherence")
    axs[0].set_ylabel("Coherence of subset")
    axs[0].set_xlim(0, 1)
    axs[0].set_ylim(0, 1)
    axs[0].axis('equal')
    plt.colorbar(sca, ax=axs[0], fraction=0.02, pad=0.04).set_label("Change time index")
    axs[1].scatter(subset_length_map[mask_cand], coherence_map[mask_cand], s=1.5,
                   c=subset_index_map[mask_cand], cmap=cmc.cm.cmaps["vik"])
    axs[1].set_xlabel("Subset length")
    axs[1].set_ylabel("Coherence of subset")
    fig_save_obj.saveAndClose("2_scatter_coherence_improvement")

    plt.figure()
    cmap = cmc.cm.cmaps["batlow"]
    cmap.set_bad(color='white')  # Set the color for masked values
    masked_data = np.ma.masked_where(change_time_map <= 0, change_time_map)
    plt.imshow(masked_data, vmin=0, vmax=slc_stack_obj.numDate, cmap=cmap, interpolation="nearest")
    plt.colorbar()
    plt.title("Change time index")
    auto_flip_direction(slc_stack_obj.metadata, ax=plt.gca(), print_msg=False)
    fig_save_obj.saveAndClose("3_change_time")

    plt.figure()
    cmap = cmc.cm.cmaps["batlow"]
    cmap.set_bad(color='white')  # Set the color for masked values
    masked_data = np.ma.masked_where(change_time_map <= 0, change_time_map)
    plt.imshow(masked_data, vmin=0, vmax=slc_stack_obj.numDate, cmap=cmap, alpha=coherence_map,
               interpolation="nearest")
    plt.colorbar()
    plt.title("Change time index with coherence overlay")
    auto_flip_direction(slc_stack_obj.metadata, ax=plt.gca(), print_msg=False)
    fig_save_obj.saveAndClose("4_change_time_w_coherence")

    plt.figure()
    cmap = cmc.cm.cmaps["batlow"]
    cmap.set_bad(color='white')  # Set the color for masked values
    masked_data = np.ma.masked_where(change_time_map <= 0, subset_length_map)
    plt.imshow(masked_data, vmin=0, vmax=max(subset_length_map.ravel()), cmap=cmap, alpha=coherence_map,
               interpolation="nearest")
    plt.colorbar()
    plt.title("Length of coherent subset with coherence overlay")
    auto_flip_direction(slc_stack_obj.metadata, ax=plt.gca(), print_msg=False)
    fig_save_obj.saveAndClose("5_length_of_coherent_subset_w_coherence")
