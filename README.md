<h1>AMDFLOW Readme</h1>

Author: Luuc Groenhart
<!--intro -->
This repository contains the codebase for the AMDFLOW model. The AMDFLOW model was created for and tested in the Master's Thesis: **"Decoding Mining's Chemical Footprint; A Proof-of-Concept of Numerical Conceptual Acid Mine Drainage Modelling for Global Scales**" at Leiden University and Technical University Delft. The model simulates Acid Mine Drainage created through the oxidation of pyrite ore at mining sites. 

<!--TOC -->
## Table of Contents
- [Install](#installation)
- [Repository Structure](#repository-structure)
- [Data](#data)
- [Usage](#usage)

<!--Install -->
## Install
The repository can be downloaded from the GitHub page and contains an environment.yaml file that can be used to recreate the environment used. See the [Conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for an explanation on how to use a .yaml (equivelent to .yml) file to create an environment.

As the codebase uses a mixture of Python and Cython code the setup and compilation of the Cython files is necessary. An setup.py file is included which should be changed depending on your system and compiler. The setup.py and .pyx files are made for compilation on a Windows 64x system with OpenMP. To run the setup.py go to your command interface, activate your environment and run: ```python setup.py build_ext --inplace```.



<!--Repo structure -->
## Repository Structure
- **case_study_I.ipynb**: Full jupyter notebook that runs a case study from data loading and integration to validating the case study.
- **case_study_II.ipynb**: See above.
- **case_study_III.ipynb**: See above.
- **case_study_IV.ipynb**: See above.
- **chemistry.pyx**: Cython .pyx file that contains the functions that run the chemistry, is called by the main AMDModel class in classes.py. After compilation the associated C/C++ files are created.
- **classes.py**: Main python file containing the AMDModel class. This class gets the dataset and runs the model by calling the functions within chemistry.pyx and transport.pyx. The model outputs .netcdf files to an output file.
- **data_utils.py**: Python file that has helper functions for data loading, integration, and transformation. Used heavily in the case study jupyter notebooks.
- **environment.yaml**: Environment file that can be used to recreate the environment used.
- **mindat_api_key.txt**: text file with your [Mindat](https://mindat.org) API key, used for Mindat API access, only necessary if new Mindat data is required.
- **mindat_global_map.py**: Python file that runs a full global data extraction from Mindat and creates a global map from it. See Data section below to read more about Mindat and how to access it's database.
- **setup.py**: Python file that runs the setup of Cython (.pyx) file compilation to C/C++. **Should be modified based on your own hard-and-software**.
- **transport.pyx**: Cython file that contains the functions that handle the transport of the AMDModel class. After compilation the associated C/C++ files are created.
- **val_plots.ipynb**: Jupyter notebook that creates validation plots, and runs the final validation calculations (note that for the final validation calculations the validation routine in val_utils.py has to be run beforehand).
- **val_utils.py**: Python file that has the full validation routine. This should be run one time per case study and saves the results per validation station - raster cell combination to a .csv. Combining the validation metrics for a full case study is done in the val_plots.ipynb.

<!-- Data -->
## Data
The AMDFLOW model needs a large amount of input data as a single xarray Dataset as input (this input into the AMDModel class). The used datasets will be shown here. 

Other data can also be used as long as they are input into the AMDModel class as an xarray Dataset with the variables: Q (time, lat, lon), ore (lat, lon), ID (lat, lon), outID (lat, lon), source (lat, lon), slope (lat, lon). These variables correspond to streamflow (Q m3/s), the amount of reactive surface area of pyrite (ore m2), unique ID per cell (ID), cell ID of where a cells flow goes to (outID), if the cell has no inflow (source), and the terrain/stream slope (slope) respectively. 

### The used data for the main model is as follows:

 - [FLO1K](https://springernature.figshare.com/collections/FLO1K_global_maps_of_mean_maximum_and_minimum_annual_streamflow_at_1_km_resolution_from_1960_through_2015/3890224): Global raster dataset at 30 arc seconds of annual average streamflow produced through machine learning methods (Barbarossa et al. 2018).

  - [HydroSHEDS con. DEM](https://www.hydrosheds.org/products/hydrosheds): Elevation data at 3 arc seconds (Lehner et al. 2008), is coarsened to 30 arc seconds in case study jupyter notebooks using bilinear resampling. Used to derive slope (see case studies).

   - [HYDROSHEDS flow direction](https://www.hydrosheds.org/products/hydrosheds): Per-cell hydrologic direction to one adjoining cell in the D8 format without any loops, at 30 arc seconds (Lehner et al. 2008). Used to create a flow network (see case studies and data_utils).

 - [Mindat](https://mindat.org): Website containing minerology data (Ma et al. 2024). Pyrite location likely to be within a mine are queried from here using the API (OpenMindat (Zhang et al. 2024)). Used in combination with Tang and Werner 2023 to estimate amount of reactive surface area of pyrite (see case studies and data_utils). **Note that an API key is needed to access the Mindat API, this key should placed in the mindat_api_key.txt file. Without this key the case study jupyter notebooks and a function in data_utils.py will cause errors if you try to query new data**

 - [Tang and Werner 2023](https://zenodo.org/records/7894216): Global polygons of all mining facilities without ores or commodities (Tang and Werner 2023). Used in combination with Mindat data to estimate amount of reactive surface area of pyrite (see case studies and data_utils). 



 ### Other Datasets used which are not 100% necessary (clipping or validation)

 - [Caravan-Qual Lite](https://zenodo.org/records/19050055): Global water quality observation dataset (Jones et al. 2025). Used for validation (comparison with simulated values).

 - [HydroRIVERS](https://www.hydrosheds.org/products/hydrorivers): Vector dataset of rivers (Lehner and Grill 2013). Used for validation (snapping output cells to rivers).

 - [HydroBASINS](https://www.hydrosheds.org/products/hydrobasins): Vector dataset of watersheds (Lehner and Grill 2013). Used to clip the input dataset to an area of interest before simulation to reduce data size.

 - [HydroSHEDS flow accumulation (acc)](https://www.hydrosheds.org/products/hydrosheds): Upstream cell amount per cell at 30 arc seconds (Lehner et al. 2008).

<!--Usage -->
## Usage
**Disclaimer: The validation of the model on the four case studies shown in the jupyter notebooks show that the model does not produce valid and reliable outputs.**

The model functions as a numerical conceptual model that calculates the transient Acid Mine Drainage pollution in a specific area of interest. The model (AMDModel in classes.py) takes an xarray Dataset as input and outputs to netcdf file. The case study jupyter notebooks perform everything from data input, data integration, data transformation, runnning the model, and validating the model. If you want to try to run a case study fully, download all the datasets and place them in a specific structure (see below), then choose your configuration options in the jupyter notebook and run the simulation. Note that certain large input datasets might take a large amount of time to run, and that runspeed is entirely hardware dependend.

The default file structure for the data, model, and output is as follows:

1. **\upperfolder\AMDFLOW**: The AMDFLOW repository should be placed in some form of upper folder.
2. **\upperfolder\data**: The HydroSHEDS con. DEM, HydroSHEDS flow direction, FLO1K, and Tang and Werner 2023 (and optionally HydroBASINS) datasets should be placed in a data file in the upper folder.
3. **\upperfolder\data\mindat_data**: Mindat data should be placed in a subfolder of the data folder.
4. **\upperfolder\data\validation data**: Validation data: Caravan-Qual Lite, HydroRIVERS, HydroSHEDS flow accumulation (acc), should be placed in a subfolder of the data folder.
5. **\upperfolder\data\validation data**: The output of the model is by default also places in the validation data subfolder.

The file structure for the data, model and output can be easily changed using keyword arguments.