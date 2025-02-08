#!/usr/bin/env python3
"""Command line interface for the spatz package."""
# SpaTZ, SPAtioTemporal Selection of Scatterers
#
# Copyright (c) 2024  Andreas Piter
# (Institute of Photogrammetry and GeoInformation, Leibniz University Hannover)
# Contact: piter@ipi.uni-hannover.de
#
# This software was developed within the context of the PhD thesis of Andreas Piter.
# The software is not yet licensed and used for internal development only.
import argparse
import json
import os
from os import getcwd
from os.path import join
import matplotlib
import numpy as np
import logging
import cmcrameri as cmc

from matplotlib import pyplot as plt
from miaplpy.objects.slcStack import slcStack
from mintpy.utils import readfile
from mintpy.utils.plot import auto_flip_direction

from sarvey.console import printCurrentConfig
from sarvey.objects import BaseStack

from spatz import change
from spatz import version
from spatz.lifetime.lifetime import runLifetimeEstimation, analyseLifetime
from spatz.utils.config import loadConfiguration, Config, generateTemplateFromConfigModel
from spatz.utils.misc import findEmptyPixels, FigureSaveHandler, setUpLogger

try:
    matplotlib.use('TkAgg')
except ImportError as e:
    print(e)


def showLogo():
    """ShowLogo."""
    logo = f"""\n
        (o>
       <__)   SpaTZ: SPAtioTemporal Scatterer Selection - v{version.__version__}
        ``\n
    """
    print(logo)


def createParser():
    """Create_parser."""
    parser = argparse.ArgumentParser(
        description='...',
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-w', '--workdir', default=None, dest="workdir",
                        help='Working directory (default: current directory).')

    parser.add_argument("-f", "--filepath", type=str, required=True, metavar="FILE",
                        help="Path to the config.json file.")

    parser.add_argument("-g", "--generate_config", action="store_true", default=False, dest="generate_config",
                        help="Write default configuration to file specified by '-f'.")

    parser.add_argument("-d", "--change_detection", action="store_true", default=False, dest="change_detection",
                        help="Run change detection.")

    # add argument for coherent lifetime estimation
    parser.add_argument("-l", "--coherent_lifetime", action="store_true", default=False, dest="coherent_lifetime",
                        help="Run coherent lifetime estimation.")

    return parser


def runChangeDetection(*, config: Config, args: argparse.Namespace, logger: logging.Logger):
    """Run the change detection.

    Identify change point for each pixel in the SLC stack.

    Parameters
    ----------
    config : Config
        Configuration object.
    args : argparse.Namespace
        Arguments from command line.
    logger : logging.Logger
        Logger object.
    """
    slc_stack_obj = slcStack(join(args.workdir, config.general.input_path, "slcStack.h5"))
    logger.info(msg="Read SLC stack.")
    slc = slc_stack_obj.read(datasetName="slc", box=None, print_msg=False)

    mask_empty = findEmptyPixels(slc=slc, meta=slc_stack_obj.metadata, logger=logger)

    logger.info(msg="Load candidate mask.")
    mask_cand = readfile.read(join(args.workdir, config.general.mask_file), datasetName='mask')[0].astype(np.bool_)

    mask_cand = mask_cand & ~mask_empty  # avoid empty pixels

    config_default_dict = generateTemplateFromConfigModel()

    method_used = None
    change_time_map = None
    if config.amplitude.run:
        method_used = "amplitude"
        printCurrentConfig(config_section=config.amplitude.dict(),
                           config_section_default=config_default_dict["amplitude"],
                           logger=logger)
        change_time_map = change.amplitude.runAmplitude(
            slc=slc,
            mask_cand=mask_cand,
            significance_level=config.amplitude.significance_level,
            logger=logger,
            num_cores=config.general.num_cores
        )
    elif config.coherence_matrix.run:
        method_used = "coherence_matrix"
        printCurrentConfig(config_section=config.coherence_matrix.dict(),
                           config_section_default=config_default_dict["coherence_matrix"],
                           logger=logger)
        change_time_map = change.coherence_matrix.runCoherenceMatrix(
            slc=slc,
            mask_cand=mask_cand,
            model_coherence=config.coherence_matrix.model_coherence,
            wdw_size=config.coherence_matrix.window_size,
            logger=logger
        )

    elif config.spatial_noise_estimation.run:
        method_used = "spatial_noise_estimation"
        printCurrentConfig(config_section=config.spatial_noise_estimation.dict(),
                           config_section_default=config_default_dict["spatial_noise_estimation"],
                           logger=logger)
        change_time_map, coherence_map, coherence_initial_map = change.phase_noise_space.runSpatialNoiseEstimation(
            slc_stack_obj=slc_stack_obj,
            slc=slc,
            mask_cand=mask_cand,
            logger=logger,
            num_cores=config.general.num_cores,
            num_link=config.spatial_noise_estimation.num_link,
            wdw_size=config.spatial_noise_estimation.window_size
        )

    elif config.temporal_noise_estimation.run:
        method_used = "temporal_noise_estimation"
        loc_inc = readfile.read(join(config.general.input_path, "geometryRadar.h5"), datasetName="incidenceAngle")[0]
        loc_inc = np.deg2rad(loc_inc)
        slant_range = readfile.read(join(config.general.input_path, "geometryRadar.h5"),
                                    datasetName="slantRangeDistance")[0]
        wavelength = float(slc_stack_obj.metadata["WAVELENGTH"])

        printCurrentConfig(config_section=config.temporal_noise_estimation.dict(),
                           config_section_default=config_default_dict["temporal_noise_estimation"],
                           logger=logger)
        change_time_map, coherence_map, coherence_initial_map = change.phase_noise_time.runTemporalNoiseEstimation(
            slc_stack_obj=slc_stack_obj,
            slc=slc,
            mask_cand=mask_cand,
            logger=logger,
            num_cores=config.general.num_cores,
            adi_p1=config.temporal_noise_estimation.adi_p1,
            adi_tcs=config.temporal_noise_estimation.adi_tcs,
            num_nearest_neighbours=config.temporal_noise_estimation.num_nearest_neighbours,
            loc_inc=loc_inc,
            slant_range=slant_range,
            wavelength=wavelength,
            demerr_bound=config.temporal_noise_estimation.dem_error_bound,
            velocity_bound=config.temporal_noise_estimation.velocity_bound,
            num_samples=config.temporal_noise_estimation.num_optimization_samples,
            show_plots=True
        )

    plt.figure()
    cmap = cmc.cm.cmaps["batlow"]
    cmap.set_bad(color='white')  # Set the color for masked values
    masked_data = np.ma.masked_where(change_time_map <= 0, change_time_map)
    plt.imshow(masked_data, vmin=0, vmax=slc_stack_obj.numDate, cmap=cmap, interpolation="nearest")
    plt.colorbar()
    plt.title("Change time index")
    auto_flip_direction(slc_stack_obj.metadata, ax=plt.gca(), print_msg=False)

    fname = join(config.general.output_path, f"change_map_{method_used}.h5")
    logger.info(msg=f"Write change map to {fname}.")
    change_map_obj = BaseStack(
        file=fname,
        logger=logger
    )

    change_map_obj.writeToFile(
        data=change_time_map,
        dataset_name="change_index",
        metadata=slc_stack_obj.metadata,
        mode="w"
    )
    plt.show()


def runCoherentLifetimeEstimation(*, config: Config, args: argparse.Namespace, logger: logging.Logger):
    """Run coherent lifetime estimation.

    Compute temporal phase coherence (Zhao and Mallorqui, 2019) for each temporal subset identified by the change
    detection method. Identify the subset with higher coherence which is assumed to be the coherent lifetime of the TCS.

    Parameters
    ----------
    config : Config
        Configuration object.
    args : argparse.Namespace
        Arguments from command line.
    logger : logging.Logger
        Logger object.
    """
    config_default_dict = generateTemplateFromConfigModel()
    printCurrentConfig(config_section=config.lifetime.dict(),
                       config_section_default=config_default_dict["lifetime"],
                       logger=logger)

    slc_stack_obj = slcStack(join(args.workdir, config.general.input_path, "slcStack.h5"))
    logger.info(msg="Read SLC stack.")
    slc = slc_stack_obj.read(datasetName="slc", box=None, print_msg=False)

    mask_empty = findEmptyPixels(slc=slc, meta=slc_stack_obj.metadata, logger=logger)

    logger.info(msg="Load candidate mask.")
    mask_cand = readfile.read(join(args.workdir, config.general.mask_file), datasetName='mask')[0].astype(np.bool_)

    mask_cand = mask_cand & ~mask_empty  # avoid empty pixels

    change_map_obj = BaseStack(file=config.lifetime.change_map_file, logger=logger)
    change_time_map = change_map_obj.read(dataset_name="change_index")

    coherence_map, coherence_initial_map, subset_length_map, subset_index_map = runLifetimeEstimation(
        slc_stack_obj=slc_stack_obj,
        slc=slc,
        mask_cand=mask_cand,
        change_time_map=change_time_map,
        logger=logger,
        num_cores=config.general.num_cores,
        wdw_size=config.lifetime.window_size,
        num_link=config.lifetime.num_link
    )

    method_used = config.lifetime.change_map_file.split("change_map_")[1].split(".h5")[0]
    path_pic = join(config.general.output_path, "pic" + "_" + method_used)
    if not os.path.exists(path_pic):
        os.makedirs(path_pic)

    fig_save_obj = FigureSaveHandler(
        path_output=path_pic, logger=logger,
        file_format="png", dpi=300, remove_ttl=False,
        save=True, close=False)

    analyseLifetime(
        coherence_initial_map=coherence_initial_map,
        coherence_map=coherence_map,
        slc_stack_obj=slc_stack_obj,
        mask_cand=mask_cand,
        change_time_map=change_time_map,
        subset_length_map=subset_length_map,
        subset_index_map=subset_index_map,
        logger=logger,
        fig_save_obj=fig_save_obj
    )

    fname = join(config.general.output_path, f"lifetime_{method_used}.h5")
    logger.info(msg=f"Write lifetime information to {fname}.")
    change_map_obj = BaseStack(
        file=fname,
        logger=logger
    )

    change_map_obj.writeToFile(data=coherence_map, metadata=slc_stack_obj.metadata, mode="w",
                               dataset_name="coherence_map")
    change_map_obj.writeToFile(data=coherence_initial_map, metadata=slc_stack_obj.metadata,
                               dataset_name="coherence_initial_map")
    change_map_obj.writeToFile(data=subset_length_map, metadata=slc_stack_obj.metadata,
                               dataset_name="subset_length_map")
    change_map_obj.writeToFile(data=subset_index_map, metadata=slc_stack_obj.metadata,
                               dataset_name="subset_index_map")
    plt.show()


def main(iargs=None):
    """Run Main."""
    parser = createParser()
    args = parser.parse_args(iargs)

    showLogo()

    if args.workdir is None:
        args.workdir = getcwd()

    print("Working directory: {}".format(args.workdir))
    plt.ion()

    config_file_path = os.path.abspath(join(args.workdir, args.filepath))

    logger = setUpLogger(path=args.workdir)

    if args.generate_config:
        logger.info(msg=f"Write default config to file: {args.filepath}.")
        default_config_dict = generateTemplateFromConfigModel()
        with open(args.filepath, "w") as f:
            f.write(json.dumps(default_config_dict, indent=4))
        return 0

    config = loadConfiguration(path=config_file_path)

    if not (config.amplitude.run or config.coherence_matrix.run or config.spatial_noise_estimation.run
            or config.temporal_noise_estimation.run):
        raise ValueError("No change detection method selected. Please select at least one method.")

    if not os.path.exists(join(args.workdir, config.general.output_path)):
        os.makedirs(join(args.workdir, config.general.output_path))

    if args.change_detection:
        logger.info(msg="Run change detection.")
        runChangeDetection(config=config, args=args, logger=logger)

    if args.coherent_lifetime:
        logger.info(msg="Run coherent lifetime estimation.")
        runCoherentLifetimeEstimation(config=config, args=args, logger=logger)

    if not (args.change_detection or args.coherent_lifetime):
        logger.info(msg="No action selected. Please select an action with '-d' or '-l'.")
    else:
        logger.info(msg="TCS detection finished normally.")

    # close log-file to avoid problems with deleting the files
    if logger.hasHandlers():
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.flush()
            handler.close()


if __name__ == '__main__':
    main()
