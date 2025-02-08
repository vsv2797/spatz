======================
Change point detection
======================

SpaTZ provides four change detection algorithms.
All methods start with a stack of coregistered SLC images and assume that only one change point is present in the time series.

The methods are based on

1. Amplitude

2. Coherence matrix

3. Phase noise, estimated with
    a. temporal displacement model

    b. spatial displacement model


The implementations in SpaTZ were used to evaluate the performance of the two methods (amplitude, coherence matrix) for transport infrastructure monitoring with Sentinel-1 InSAR (Piter et al., 2025).

The methods are implemented in the `spatz.change` module.


Amplitude
---------

The method based on the amplitude time series was proposed by Hu et al. (2019).
It was again used by Dörr et al. (2022), but extended with a phase refinement step (see 'Phase noise').


Coherence matrix
----------------

The method based on coherence matrix was proposed by Monti-Guarnieri et al. (2018), but initially designed for change detection in SAR stacks and applied to 18 images.
For larger stacks, the computation time is much higher compared to the other change detection methods.


Phase noise
-----------

The method based on the phase noise was initially proposed by Dörr et al. (2022).
Beside the temporal displacement model, a spatial displacement model is also implemented in SpaTZ.


Temporal displacement model
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The original method developed by Dörr et al. (2022) is based on a temporal displacement model.
The implementation in SpaTZ does not use the MSBAS approach but instead temporally unwraps a single-reference network to estimate the phase noise.
Moreover, the implementation in SpaTZ does not use a pre-selection of TCS candidates with the amplitude-based method, but instead processes all pixel below a given amplitude dispersion threshold.


Spatial displacement model
~~~~~~~~~~~~~~~~~~~~~~~~~~
The author of this package (A. Piter) proposes a method based on a spatial displacement model.
This method assumes spatial smoothness of the displacement signal and estimates the phase noise with the method proposed by Zhao & Mallorqui (2019).


==========
Literature
==========

* Monti-Guarnieri, A. V.; Brovelli, M. A.; Manzoni, M.; Mariotti d'Alessandro, M.; Molinari, M. E. & Oxoli, D. (2018). Coherent Change Detection for Multipass SAR. IEEE Transactions on Geoscience and Remote Sensing, 2018, 56, 6811-6822.

* Hu, F.; Wu, J.; Chang, L. & Hanssen, R. F. (2019). Incorporating Temporary Coherent Scatterers in Multi-Temporal InSAR Using Adaptive Temporal Subsets. IEEE Transactions on Geoscience and Remote Sensing, 2019, 57, 7658-7670.

* Zhao, F. & Mallorqui, J. J. (2019). A Temporal Phase Coherence Estimation Algorithm and Its Application on DInSAR Pixel Selection. IEEE Transactions on Geoscience and Remote Sensing, 2019, 57, 8350-8361.

* Dörr, N.; Schenk, A. & Hinz, S. (2022). Fully Integrated Temporary Persistent Scatterer Interferometry. IEEE Transactions on Geoscience and Remote Sensing, 2022, 60, 1-15.

* Piter, A.: Haghighi, M. H.; Motagh, M. (2025). Temporary Coherent Scatterer Selection for Transport Infrastructure Monitoring with Sentinel-1 InSAR. 6h Joint International Symposium on Deformation Monitoring (JISDM), 2025, submitted.

