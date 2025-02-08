"""Miscellaneous functions for TCS processing."""
# SpaTZ, SPAtioTemporal Selection of Scatterers
#
# Copyright (c) 2024  Andreas Piter
# (Institute of Photogrammetry and GeoInformation, Leibniz University Hannover)
# Contact: piter@ipi.uni-hannover.de
#
# This software was developed within the context of the PhD thesis of Andreas Piter.
# The software is not yet licensed and used for internal development only.
import os
import sys
import time
from os.path import join

import numpy as np
import cmcrameri as cmc
from matplotlib import pyplot as plt
import logging

from mintpy.utils.plot import auto_flip_direction


def findEmptyPixels(*, slc: np.ndarray, meta: dict, logger: logging.Logger) -> np.ndarray:
    """Find empty pixels in the image and return a mask of valid pixels.

    Parameters
    ----------
    slc : np.ndarray
        Complex image.
    meta : dict
        Metadata of the image.
    logger : logging.Logger
        Logger object.

    Returns
    -------
    np.ndarray
        Mask of valid pixels.
    """
    logger.info(msg="Find empty pixels in the image.")
    mask_empty = np.zeros(slc.shape[1:], dtype=np.bool_)
    for i in range(slc.shape[0]):
        mask_empty |= slc[i, :, :] == 0
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(mask_empty, cmap=cmc.cm.cmaps["grayC"], interpolation="nearest")
    axs[0].set_xlabel("range")
    axs[0].set_ylabel("azimuth")
    axs[0].set_title("Empty pixels in at least one image")
    auto_flip_direction(meta, ax=axs[0], print_msg=False)

    mean_amp_img = np.mean(np.abs(slc), axis=0)
    axs[1].imshow(mean_amp_img == 0, cmap=cmc.cm.cmaps["grayC"], interpolation="nearest")
    axs[1].set_xlabel("range")
    axs[1].set_ylabel("azimuth")
    axs[1].set_title("Empty pixels in all images")
    auto_flip_direction(meta, ax=axs[1], print_msg=False)

    logger.info(msg=f"Empty pixels found in the image: {np.sum(~mask_empty)}")
    plt.show()
    return mask_empty


class FigureSaveHandler:
    """FigureSaveHandler."""

    def __init__(self, *, path_output: str, logger: logging.Logger, file_format: str, save: bool,
                 remove_ttl: bool, close: bool, dpi: int = 300):
        """FigureSaveHandler.

        Parameters
        ----------
        path_output : str
            Path to save the figure.
        logger : logging.Logger
            Logger object.
        file_format : str
            Format of the figure (e.g. png).
        save : bool
            Save the figure.
        remove_ttl : bool
            Remove the title from the plot before saving.
        close : bool
            Close the plot after saving.
        dpi : int
            Resolution of the figure in dots per inch (default: 300).
        """
        self.logger = logger
        self.path_output = path_output
        self.file_format = file_format
        self.remove_ttl = remove_ttl
        self.save = save
        self.close = close
        self.dpi = dpi

    def saveAndClose(self, fname: str):
        """SaveFigAndClose."""
        if self.save:
            path_file = join(self.path_output, fname + "." + self.file_format)
            self.logger.info(msg=f"Save figure: {path_file}")
            if self.remove_ttl:
                plt.gca().set_title("")
            plt.gcf().savefig(path_file, dpi=self.dpi, format=self.file_format, bbox_inches="tight")
            if self.close:
                plt.close(plt.gcf())


def setUpLogger(*, path: str):
    """SetUpLogger."""
    # initiate logger
    logging_level = logging.getLevelName('DEBUG')  # set a default value until level is read from config

    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    current_datetime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_filename = f"log_{current_datetime}.log"
    if not os.path.exists(join(path, "logfiles")):
        os.mkdir(join(path, "logfiles"))
    file_handler = logging.FileHandler(filename=join(path, "logfiles", log_filename))
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    logger.setLevel(logging_level)
    return logger
