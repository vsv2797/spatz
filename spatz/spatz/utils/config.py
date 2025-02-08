"""Configuration module for TCS detection."""
# SpaTZ, SPAtioTemporal Selection of Scatterers
#
# Copyright (c) 2024  Andreas Piter
# (Institute of Photogrammetry and GeoInformation, Leibniz University Hannover)
# Contact: piter@ipi.uni-hannover.de
#
# This software was developed within the context of the PhD thesis of Andreas Piter.
# The software is not yet licensed and used for internal development only.
import os
import json
from json import JSONDecodeError

from pydantic import BaseModel, Field, validator, Extra
from pydantic.schema import schema


class General(BaseModel, extra=Extra.forbid):
    """General processing settings."""

    input_path: str = Field(
        title="The path to the input data directory.",
        description="The input data directory must contain 'slcStack.h5' and 'geometryRadar.h5' created with MiaplPy.",
        default="inputs/"
    )

    output_path: str = Field(
        title="The path to the processing output data directory.",
        description="The resulting change time map and coherent lifetime estimation will be stored in this directory.",
        default="outputs_tcs/"
    )

    mask_file: str = Field(
        title="The path to the mask file.",
        description="The mask file is used to mask out areas for TCS processing. Only pixels within the mask will be"
                    "evaluated.",
        default="mask.h5"
    )

    num_cores: int = Field(
        title="Number of cores",
        description="Set the number of cores for parallel processing.",
        default=30
    )

    @validator('input_path')
    def checkPathInputs(cls, v):
        """Check if the input path exists."""
        if v == "":
            raise ValueError("Empty string is not allowed.")
        if not os.path.exists(os.path.abspath(v)):
            raise ValueError(f"input_path is invalid: {os.path.abspath(v)}")
        if not os.path.exists(os.path.join(os.path.abspath(v), "slcStack.h5")):
            raise ValueError(f"'slcStack.h5' does not exist: {v}")
        if not os.path.exists(os.path.join(os.path.abspath(v), "geometryRadar.h5")):
            raise ValueError(f"'geometryRadar.h5' does not exist: {v}")
        return v

    @validator('mask_file')
    def checkMaskFile(cls, v):
        """Check if the mask file exists."""
        if v == "":
            raise ValueError("Empty string is not allowed.")
        if not os.path.exists(os.path.abspath(v)):
            raise ValueError(f"mask_file is invalid: {v}")
        return v

    @validator('num_cores')
    def checkNumCores(cls, v):
        """Check if the number of cores is valid."""
        if v <= 0:
            raise ValueError("Number of cores must be greater than zero.")
        return v


class Amplitude(BaseModel, extra=Extra.forbid):
    """Settings amplitude change detection."""

    run: bool = Field(
        title="Run amplitude change detection",
        description="Set to true to run the amplitude change detection.",
        default=False
    )

    significance_level: float = Field(
        title="Significance level of hypothesis test",
        description="Set the significance level for change point detection.",
        default=0.0001
    )

    @validator('significance_level')
    def checkSignificanceLevel(cls, v):
        """Check if the alpha value is valid."""
        if v <= 0:
            raise ValueError("Significance level alpha must be greater than zero.")
        return v


class CoherenceMatrix(BaseModel, extra=Extra.forbid):
    """Settings for coherence matrix change detection."""

    run: bool = Field(
        title="Run coherence matrix change detection",
        description="Set to true to run the coherence matrix change detection.",
        default=False
    )

    model_coherence: float = Field(
        title="Coherence value for modelling the coherence matrix.",
        description="A coherence matrix is modelled with the given coherence value to estimate the change time."
                    "The coherence is used in equations 10 and 17 (Monti-Guarnieri et al, 2018, IEEE TGRS) to model the"
                    "coherence matrix.",
        default=0.5
    )

    window_size: int = Field(
        title="Size of window [pixel]",
        description="The coherence matrix is estimated from a set of neighbouring pixels which are within a window "
                    "centered at the pixel.",
        default=9
    )

    @validator('model_coherence')
    def checkModelCoherence(cls, v):
        """Check if the model_coherence value is valid."""
        if (v < 0) or (v > 1):
            raise ValueError("Coherence must be within interval (0, 1).")
        return v

    @validator('window_size')
    def checkWindowSize(cls, v):
        """Check if the window size is valid."""
        if v <= 0:
            raise ValueError("Filter window size must be greater than zero.")
        return v


class SpatialNoiseEstimation(BaseModel, extra=Extra.forbid):
    """Settings for spatial noise estimation change detection."""

    run: bool = Field(
        title="Run spatial noise estimation change detection",
        description="Set to true to run the change detection based on the phase noise computed with a spatial "
                    "displacement model.",
        default=False
    )

    window_size: int = Field(
        title="Size of window [pixel]",
        description="Set the size of window for estimating the spatially correlated phase contributions from the"
                    "neighbourhood of the pixel. Details can be found in Zhao and Mallorqui (2019), IEEE TGRS.",
        default=9
    )

    num_link: int = Field(
        title="Number of links/interferograms between images in the interferogram network",
        description="Set the number of links for small baseline interferogram network. Increasing the number of links"
                    "will increase the computational time as the phase noise has to be estimated for each "
                    "interferogram.",
        default=5
    )

    @validator('window_size')
    def checkWindowSize(cls, v):
        """Check if the window size is valid."""
        if v <= 0:
            raise ValueError("Filter window size must be greater than zero.")
        return v

    @validator('num_link')
    def checkNumLink(cls, v):
        """Check if the number of links is valid."""
        if v <= 0:
            raise ValueError("Num_link must be greater than zero.")
        return v


class TemporalNoiseEstimation(BaseModel, extra=Extra.forbid):
    """Settings for temporal noise estimation change detection."""

    run: bool = Field(
        title="Run temporal noise estimation change detection",
        description="Set to true to run the temporal noise estimation change detection.",
        default=False
    )

    adi_p1: float = Field(
        title="Threshold for amplitude dispersion for first-order point selection.",
        description="Set the threshold for amplitude dispersion for first-order point selection. The amplitude "
                    "dispersion holds as a good approximation for the phase noise only for values below 0.2.",
        default=0.2
    )

    adi_tcs: float = Field(
        title="Threshold for amplitude dispersion for TCS candidate selection.",
        description="Set the threshold for amplitude dispersion for TCS candidate selection. A higher threshold "
                    "compared to the first-order point selection is recommended to include more potential TCS"
                    "candidates. However, more noisy pixels will be included and the number of false alarms will"
                    "increase.",
        default=0.4
    )

    num_nearest_neighbours: int = Field(
        title="Number of nearest neighbours for first-order point search.",
        description="The phase noise of a TCS candidate is evaluated w.r.t. a virtual reference point. The virtual"
                    "reference point is the mean of the phase values of the k-nearest first-order points of the TCS"
                    "candidate.",
        default=5
    )

    velocity_bound: float = Field(
        title="Bounds on mean velocity for temporal unwrapping [m/year]",
        description="Set the bound (symmetric) for the mean velocity in temporal unwrapping.",
        default=0.15
    )

    dem_error_bound: float = Field(
        title="Bounds on DEM error for temporal unwrapping [m]",
        description="Set the bound (symmetric) for the DEM error estimation in temporal unwrapping.",
        default=100.0
    )

    num_optimization_samples: int = Field(
        title="Number of samples in the search space for temporal unwrapping",
        description="Set the number of samples evaluated along the search space for temporal unwrapping.",
        default=100
    )

    @validator('adi_p1', 'adi_tcs')
    def checkADIP1(cls, v):
        """Check if the threshold for adi is valid."""
        if v <= 0:
            raise ValueError('Threshold for amplitude dispersion must be greater than zero.')
        return v

    @validator('num_nearest_neighbours')
    def checkKNN(cls, v):
        """Check if the k-nearest neighbours is valid."""
        if v <= 0:
            raise ValueError('Number of nearest neighbours cannot be negative or zero.')
        return v

    @validator('velocity_bound')
    def checkVelocityBound(cls, v):
        """Check if the velocity bound is valid."""
        if v <= 0:
            raise ValueError('Velocity bound cannot be negative or zero.')
        return v

    @validator('dem_error_bound')
    def checkDEMErrorBound(cls, v):
        """Check if the DEM error bound is valid."""
        if v <= 0:
            raise ValueError('DEM error bound cannot be negative or zero.')
        return v

    @validator('num_optimization_samples')
    def checkNumSamples(cls, v):
        """Check if the number of samples for the search space is valid."""
        if v <= 0:
            raise ValueError('Number of optimization samples cannot be negative or zero.')
        return v


class Lifetime(BaseModel, extra=Extra.forbid):
    """Settings for coherent lifetime estimation."""

    run: bool = Field(
        title="Run coherent lifetime estimation",
        description="Set to true to run the coherent lifetime estimation.",
        default=False
    )

    change_map_file: str = Field(
        title="The path to the change map file.",
        description="Set the path of the change map file.",
        default="change_map.h5"
    )

    window_size: int = Field(
        title="Size of window [pixel]",
        description="Set the size of window for coherence estimation with TPC.",
        default=9
    )

    num_link: int = Field(
        title="Number of links/interferograms between images in the interferogram network",
        description="Set the number of links for small baseline interferogram network.",
        default=5
    )

    @validator('window_size')
    def checkWindowSize(cls, v):
        """Check if the window size is valid."""
        if v <= 0:
            raise ValueError("Filter window size must be greater than zero.")
        return v

    @validator('num_link')
    def checkNumLink(cls, v):
        """Check if the number of links is valid."""
        if v <= 0:
            raise ValueError("Num_link must be greater than zero.")
        return v

    # @validator('change_map_file')
    # def checkChangeMapFile(cls, v):
    #     """Check if the change map file exists."""
    #     if v == "":
    #         raise ValueError("Empty string is not allowed.")
    #     if not os.path.exists(os.path.abspath(v)):
    #         raise ValueError(f"change_map_file is invalid: {v}")
    #     return v


class Config(BaseModel):
    """Configurations for Temporary coherent scatterer detection."""

    general: General = Field(
        title="General", description="General settings for TCS detection."
    )

    amplitude: Amplitude = Field(
        title="Amplitude", description="Settings for amplitude change detection."
    )

    coherence_matrix: CoherenceMatrix = Field(
        title="CoherenceMatrix", description="Settings for coherence matrix change detection."
    )

    spatial_noise_estimation: SpatialNoiseEstimation = Field(
        title="SpatialNoiseEstimation", description="Settings for spatial noise estimation change detection."
    )

    temporal_noise_estimation: TemporalNoiseEstimation = Field(
        title="TemporalNoiseEstimation", description="Settings for temporal noise estimation change detection."
    )

    lifetime: Lifetime = Field(
        title="Lifetime", description="Settings for coherent lifetime estimation."
    )


def loadConfiguration(*, path: str):
    """Load configuration json file.

    Parameters
    ----------
    path : str
        Path to the configuration json file.

    Returns
    -------
    : dict
        A dictionary containing configurations.

    Raises
    ------
    JSONDecodeError
        If failed to parse the json file to the dictionary.
    FileNotFoundError
        Config file not found.
    IOError
        Invalid JSON file.
    ValueError
        Invalid value for configuration object.
    """
    try:
        with open(path) as config_fp:
            config = json.load(config_fp)
            config = Config(**config)
    except JSONDecodeError as e:
        raise IOError(f'Failed to load the configuration json file => {e}')
    return config


def generateTemplateFromConfigModel():
    """GenerateTemplateFromConfigModel."""
    top_level_schema = schema([Config])
    top_level_dict = dict()
    for sec_name, sec_def in top_level_schema['definitions'].items():
        if sec_name == "Config":
            # substitute the class names of subsections in top_level_dict by the name of the sections in class Config
            for subsec_name, subsec_def in sec_def["properties"].items():
                top_level_dict[subsec_name] = top_level_dict.pop(subsec_def["title"])
            continue  # don't add "Config" to top_level_dict
        sec_dict = dict()
        for subsec_name, subsec_def in sec_def["properties"].items():
            if "default" not in subsec_def:
                sec_dict.update({subsec_name: None})
            else:
                sec_dict.update({subsec_name: subsec_def["default"]})
        top_level_dict.update({sec_name: sec_dict})

    return top_level_dict
