SpaTZ - SPAtioTemporal Selection of Scatterers
==============================================

SpaTZ is a tool for the selection of scatterers for InSAR time series analysis.
Scatterers are selected based on assumptions on spatial and/or temporal characteristics of the radar signal.
This repository is part of current research and is subject to change.


Installation
------------

This repository uses the same environment as `SARvey <https://github.com/luhipi/sarvey>`_.
Follow the instructions in SARvey.
Afterwards, install SpaTZ with the following command:

.. code-block:: bash

    mamba activate sarvey
    cd spatz
    pip install .

Feature overview
----------------

**SpaTZ** is a command line tool that provides two features:

1. **Change detection** from a stack of coregistered SLC images

2. **Coherent lifetime** estimation based on the change detection results


Usage
-----

Create a default configuration file in the working directory with `-g`:

.. code-block:: bash

    spatz -f config_spatz.json -g

Create a mask file with the area of interest within the radar image.
For monitoring transport infrastructure, you can e.g. use the scripts provided in the **SARvey** repository.
A description of how to create a mask from OpenStreetMap data with `sarvey_osm` and `sarvey_mask` can be found in the **SARvey** documentation.
Alternatively, the mask can be created manually with the `generate_mask.py` script from **MintPy**.
You can manually draw a polygon e.g. on top of the height image:

.. code-block:: bash

    generate_mask.py inputs/geometryRadar.h5 height -o mask.h5 --roipoly

Next, open the configuration file with a text editor, select one change detection method and adjust the corresponding parameters.

Run the change detection with the `-d` (detection) option:

.. code-block:: bash

    spatz -f config_spatz.json -d

After the change detection, the coherent lifetime can be estimated for each pixel in the image.
Adjust the parameters in the configuration file for the coherent lifetime estimation.
Estimate the coherent lifetime with the `-l` (lifetime) option:

.. code-block:: bash

    spatz -f config_spatz.json -l

Credits
-------

This repository is developed by Andreas Piter (piter@ipi.uni-hannover.de) within his research at `Institute of Photogrammetry and GeoInformation`_, Leibniz University Hannover.
Mahmud H. Haghighi (mahmud@ipi.uni-hannover.de) and Mahdi Motagh (motagh@gfz-potsdam.de) supervised this work.


.. _`Institute of Photogrammetry and GeoInformation`: https://www.ipi.uni-hannover.de/en/

