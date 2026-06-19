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

<!-- Data -->
## Data

<!--Usage -->
## Usage