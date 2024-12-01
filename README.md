# RADNN

*Last edited: 2024-11-06*

This repository contains my research related to the use of physics-based machine learning (PIML) in climate and climatic models schemes. It is mainly based on the works of Ukkonen et al. (see [References](#References)). One of the researches being done is to understand the use of PIML to emulate the gas optics scheme lookup table [RRTMGP](https://github.com/earth-system-radiation/rte-rrtmgp).

This repository is a work in progress and is subject to constant change.

## Notebooks

(unordered)

- [rfmip02-clear-sky-NN-02.ipynb](rfmip02-clear-sky-NN.ipynb) : version 2.

- [rfmip02-clear-sky-02.ipynb](rfmip02-clear-sky.ipynb) : version 2.

- [rfmip02-clear-sky-NN.ipynb](rfmip02-clear-sky-NN.ipynb) : "RRTMG-NN". Runs the RFMIP-CLEAR-SKY example with NN, from ukk22test02 dir. Was "ukk22test02-rfmip-clear-sky-NN.ipynb".

- [rfmip02-clear-sky.ipynb](rfmip02-clear-sky.ipynb) : "RRTMG". Runs the RFMIP-CLEAR-SKY example without NN, from ukk22test02 dir. Was "ukk22test02-rfmip-clear-sky.ipynb".

- [rrtmgp_rfmip_lw-test01.ipynb](rrtmgp_rfmip_lw-test01.ipynb) : example program to demonstrate the calculation of longwave radiative fluxes in clear, aerosol-free skies. Based on `rfmip-clear-sky/rrtmgp_rfmip_lw.F90` from git branch `main`.

- [ukk23test01-train-v2.ipynb](ukk23test01-train-v2.ipynb) : continuation of `ukk23test01-train-v1.ipynb`, adding more documentation, better organization, complete training, etc.

- [ukk23test01-train-v1.ipynb](ukk23test01-train-v1.ipynb) : generates files containing the NN model that is later used in the RRTMGP-NN model. The implementation uses TensorFlow/Python for training the NN, and Fortran routines are used to generate the training dataset. Based on [[Ukk22]](#ref01) git main branch.

- [rfmip01-clear-sky.ipynb](rfmip01-clear-sky.ipynb) : runs the RFMIP-CLEAR-SKY example, from ukk23test01 dir, described in [Ukk22]. Was "ukk23test01-rfmip-clear-sky.ipynb".

- [ecrad01-NN-gprof.ipynb](ecrad01-NN-gprof.ipynb) : "ecRad-NN". Gprof of ecRad from `ukk23eo01` dir (uses RRTMGP-NN). Was "ukk23eo01-NN-gprof.ipynb".

- [ecrad01-gprof.ipynb](ecrad01-gprof.ipynb) : "ecRad". Gprof of ecrad (without NN) executable from ecrad dir.

- [ecrad-01-sd-v240823.ipynb](ecrad-01-sd-v240823.ipynb) : shows the original ecRad radiation module using conventional numerical method, running on SDumont.

- [ukk23test01-train-sd-v240823.ipynb](ukk23test01-train-sd-v240823.ipynb) : NN network training for the optical gas radiation problem, running on SDumont.

## Directories

- [ukk22test02](ukk22test02) : git branch `nn_dev`, described in [Ukk22c]. Data: <https://zenodo.org/records/5833494>

- [ukk23test01](ukk23test01) : git branch `main`, described in [Ukk22]. Contains the sub-dir `/examples/rrtmgp-nn-training` with the implementation for training gas optics NN.

- [ukk23eo01](ukk23eo01) : "ecRad-NN". Contains the optimized version of the ecRad radiation scheme, with the new RRTMGP-NN gas optics (see ukk23test01). Does not contain the implementation that does the NN training. The implementation, development and testing of RRTMGP-NN is described in [Ukk23]. Sources:
  
     - <https://github.com/peterukk/ecrad-opt> : "(...) the most up-to-date optimized ecRad code, see branch `clean_no_opt_testing` in this github repo (...)"

- [ecrad](ecrad) : original ecRad repo, without NN
  
     - <https://github.com/ecmwf-ifs/ecrad>

## Files

- [ecrad-radiation-user-guide-2022.md](ecrad-radiation-user-guide-2022.md) : ecRad Radiation Scheme User Guide original document converted from PDF to Markdown:
  
     - Hogan, R. J. [ecRad radiation scheme: User Guide](https://confluence.ecmwf.int/download/attachments/70945505/ecrad_documentation.pdf?version=5&modificationDate=1655480733414&api=v2). Version 1.5 (June 2022) applicable to ecRad version 1.5.x .

- `*.txt` and `*.yml` are auxiliary files.

- `*.nc` files are of type NetCDF4 and can be browsed and their structure visualized using the Java tool [ToolsUI](https://docs.unidata.ucar.edu/netcdf-java/current/userguide/reading_cdm.html) or using the Python library [netcdf4-python](https://github.com/Unidata/netcdf4-python).

- [env.md](env.md) :  briefly describes the installation of the Conda environment used in Notebooks.

## Code and data

Due to size restrictions, the data is not present in this repository and must be obtained and installed from several sources:

- <https://zenodo.org/records/7413935>
- <https://zenodo.org/records/7413952>
- <https://zenodo.org/records/7852526>
- <https://zenodo.org/records/4030436>
- <https://zenodo.org/records/5833494>
- <https://github.com/peterukk/rte-rrtmgp-nn>

The repositories in <https://github.com/peterukk> contain code and some data, distributed across different repo branches. The RTE+RRTMGP-NN is available on

- <https://github.com/peterukk/rte-rrtmgp-nn> (there are multiple branches in the repository containing different data files).

- <https://doi.org/10.5281/zenodo.7413935> [Ukk22]
  
     - Redirects to: <https://zenodo.org/records/7413935> . "peterukk/rte-rrtmgp-nn: 2.0".

- The Fortran programs and Python scripts used for data generation and model training are found in the directory `examples/rrtmgp-nn-training`.

The training data and archived version of RTE+RRTMGP-NN 2.0 with its training scripts can be accessed at

- <https://doi.org/10.5281/zenodo.6576680> (see [Ukk22d])
  
     - Redirects to: <https://zenodo.org/records/7413952> .  "Code and extensive data for training neural networks for radiation, used in "Implementation of a machine-learned gas optics parameterization in the ECMWF Integrated Forecasting System: RRTMGP-NN 2.0" ".

The optimized version of the ecRad radiation scheme integrated with RRTMGP-NN 2.0 can be accessed at

- <https://doi.org/10.5281/zenodo.7148329> (see [Ukk22a])
  
     - Redirects to: <https://zenodo.org/records/7852526> . "Optimized version of the ecRad radiation scheme with new RRTMGP-NN gas optics".

[Ukk21] training and evaluation data for machine learning models emulating the RTE+RRTMGP radiation scheme or its components.

- <https://doi.org/10.5281/zenodo.5833494>
  
     - Redirects to: <https://zenodo.org/records/5833494> . "Training and evaluation data for machine learning models emulating the RTE+RRTMGP radiation scheme or its components.".

## Random notes

1. The Table of Contents, for Markdown and Notebook files, can be viewed in JupyterLab using the "View > Table of Contents" menu item or by pressing Ctrl+Shift+K.

2. From <https://github.com/peterukk/rte-rrtmgp-nn/tree/2.0> :
   
      1. Instead of the original lookup-table interpolation routine and "eta" parameter to handle the overlapping absorption of gases in a given band, this fork implements neural networks (NNs) to predict optical properties for given atmospheric conditions and gas concentrations, which includes all minor longwave (LW) gases supported by RRTMGP. The NNs predict molecular absorption (LW/SW), scattering (SW) or emission (LW) for all spectral points from an input vector consisting of temperature, pressure and gas concentrations of an atmospheric layer. The models have been trained on 6-7 million samples (LW) spanning a wide range of conditions (pre-industrial, present-day, future...) so that they may be used for both weather and climate applications.

3. From <https://zenodo.org/records/7413952> :
   
      1. The files contain datasets for training neural network versions of the RRTMGP gas optics scheme (as described in the paper) that are read by `ml_train.py`.
   
      2. The ML datasets were generated using the input profiles datasets and running the Fortran programs `rrtmgp_sw_gendata_rfmipstyle.F90` and `rrtmgp_lw_gendata_rfmipstyle.F90` in `rte-rrtmgp-nn/examples/rrtmgp-nn-training`, which call the RRTMGP gas optics scheme.

4. From [Ukk20] :
   
      - The GPTL profiler was used to profile the code. <https://jmrosinski.github.io/GPTL/>
   
      - Also used to obtain data: RFMIP (Radiative Forcing Model Intercomparison Project).
   
      - RTE+RRTMGP-NN is available on Github. <https://github.com/peterukk/rte-rrtmgp-nn>
   
      - Version archived online: <https://doi.org/10.5281/zenodo.4029138>
        
           - Is supplement to
             <https://github.com/peterukk/rte-rrtmgp-nn/tree/0.9>
   
      - Scripts and data used in this paper are available online: <https://doi.org/10.5281/zenodo.3909653>

5. The NN inference and I/O code in RRTGMP-NN is based on Neural-Fortran  [Cur19].

6. **RTE+RRTMGP** [Pin19] is a recently developed **radiation transfer scheme** for dynamical models combining two codes: Radiative Transfer for Energetics (RTE), which computes fluxes given a description of boundary conditions, source functions and optical properties of the atmosphere, and RRTM for General circulation model applications — Parallel (RRTMGP), which computes optical properties and source functions of the gaseous atmosphere. The **gas optics scheme RRTMGP** uses a k-distribution based on state-of-the-art spectroscopy, and has 256 g-points in the longwave and 224 g-points in the shortwave, which is high compared to many other schemes. [Ukk20]

## Main references

- [Ukk22] Ukkonen, P., et al. (Dec 8, 2022). "peterukk/rte-rrtmgp-nn: 2.0". (see [Ukk23]) (**code**) (includes NN training)
  
     - <https://doi.org/10.5281/zenodo.7413935> (redirects to <https://zenodo.org/records/7413935>)
  
     - <https://github.com/peterukk/rte-rrtmgp-nn/tree/2.0>

- [Ukk22d] Ukkonen, P. (2022). Code and extensive data for training neural networks for radiation, used in “Implementation of a machine-learned gas optics parameterization in the ECMWF Integrated Forecasting System: RRTMGP-NN 2.0”" [Dataset].
  
     - <https://doi.org/10.5281/zenodo.7413952> (redirects to <https://zenodo.org/records/7413952>)

- [Ukk22a] Ukkonen, P. (Oct 5, 2022). Optimized version of the ecRad radiation scheme with new RRTMGP-NN gas optics. (code and data) (does not include the NN training)
  
     - <https://doi.org/10.5281/zenodo.7852526> (redirects to <https://zenodo.org/records/7852526>)
  
     - Code: <https://github.com/peterukk/ecrad-opt/tree/clean_no_opt_testing>

- [Ukk22c] Ukkonen, P. (2022). Exploring Pathways to More Accurate Machine Learning Emulation of Atmospheric Radiative Transfer. Journal of Advances in Modeling Earth Systems, *14*(4), e2021MS002875. <https://doi.org/10.1029/2021MS002875>

- [Ukk23] Ukkonen, P., & Hogan, R. J. (2023). Implementation of a machine-learned gas optics parameterization in the ECMWF Integrated Forecasting System: RRTMGP-NN 2.0. Geoscientific Model Development, 16(11), 3241–3261.
  
     - <https://doi.org/10.5194/gmd-16-3241-2023> (redirects to <https://gmd.copernicus.org/papers/16/3241/2023>)
  
     - The RRTMGP-NN scheme is described in [Ukk20].

- [Ukk21] Ukkonen, P. (2021). Training and evaluation data for machine learning models emulating the RTE+RRTMGP radiation scheme or its components.
  
     - <https://doi.org/10.5281/zenodo.5833494>

- [Ukk20] Ukkonen, P., Pincus, R., Hogan, R. J., Pagh Nielsen, K., & Kaas, E. (2020). Accelerating Radiation Computations for Dynamical Models With Targeted Machine Learning and Code Optimization. Journal of Advances in Modeling Earth Systems, 12(12).
  
     - <https://doi.org/10.1029/2020MS002226> (redirects to <https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020MS002226>)
  
     - <https://zenodo.org/records/4030436>. Supplementary Python code and data used to train NN.
  
     - <https://zenodo.org/records/4029138> (RRTMGP-NN 2020 code and data. Is supplement to <https://github.com/peterukk/rte-rrtmgp-nn/tree/0.9>)

- [Cur19] Curcic, M. (2019). A parallel Fortran framework for neural networks and deep learning. *ACM SIGPLAN Fortran Forum*, *38*(1), 4–21.
  
     - <https://doi.org/10.1145/3323057.3323059>

- [Pin19] Pincus, R., Mlawer, E. J., & Delamere, J. S. (2019). Balancing Accuracy, Efficiency, and Flexibility in Radiation Calculations for Dynamical Models. Journal of Advances in Modeling Earth Systems, *11*(10), 3074–3089.  (includes Neural-Fortran)
  
     - <https://doi.org/10.1029/2019MS001621>
  
     - <https://github.com/RobertPincus/rte-rrtmgp>
