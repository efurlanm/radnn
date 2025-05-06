# RADNN

This repository houses my research notes on physics-based machine learning (PIML) applied to scientific problems, including experiments in climate and weather modeling. As the research is actively progressing, this repository remains a work in progress (WIP) and is subject to change.

## Some directories

- [ukk23test01](ukk23test01/README.md) - tests performed on a clone of the **main** git branch described in [[Ukk22]](references.md#Ukk22): <https://github.com/peterukk/rte-rrtmgp-nn/tree/2.0> . Includes the `examples/rrtmgp-nn-training` subdirectory with the implementation for NN gas optics training.

- [ukk22test02](ukk22test02/README.md) - tests performed on a clone of the **nn_dev** git branch described in [[Ukk22c]](references.md#Ukk22c): <https://github.com/peterukk/rte-rrtmgp-nn/tree/nn_dev/> .  Includes the `examples/rrtmgp-nn-training` subdirectory with the implementation for NN gas optics training.

- [ukk23eo01](ukk23eo01/README.md) - "ecRad-NN". Contains the optimized version of the ecRad radiation scheme, with the new RRTMGP-NN gas optics, from the `clean_no_opt_testing` branch of <https://github.com/peterukk/ecrad-opt>, described in [[Ukk22a]](references.md#Ukk22a). It does not contain the implementation that performs NN training (the implementation, development and testing of RRTMGP-NN are described in [[Ukk23]](references.md#Ukk23)).

- [ecrad](ecrad/README.md) - original ecRad repo, without NN. <https://github.com/ecmwf-ifs/ecrad> .

- [torchfort](torchfort/README.md) - tests performed on a clone of the Nvidia TorchFort git repository <https://github.com/NVIDIA/TorchFort>

## Notebooks

(unordered)

- [rfmip02-clear-sky-NN-02.ipynb](rfmip02-clear-sky-NN.ipynb) - junk copy used for various tests.

- [rfmip02-clear-sky-02.ipynb](rfmip02-clear-sky.ipynb) - junk copy used for various tests.

- [rfmip02-clear-sky-NN.ipynb](rfmip02-clear-sky-NN.ipynb) - "RRTMG-NN". Runs the RFMIP-CLEAR-SKY example with NN, from ukk22test02 dir.

- [rfmip02-clear-sky.ipynb](rfmip02-clear-sky.ipynb) - "RRTMG". Runs the RFMIP-CLEAR-SKY example without NN, from ukk22test02 dir.

- [rrtmgp_rfmip_lw-test01.ipynb](rrtmgp_rfmip_lw-test01.ipynb) - example program to demonstrate the calculation of longwave radiative fluxes in clear, aerosol-free skies. Based on `rfmip-clear-sky/rrtmgp_rfmip_lw.F90` from git branch **main**.

- [ukk23test01-train-v2.ipynb](ukk23test01-train-v2.ipynb) - continuation of `ukk23test01-train-v1.ipynb`, adding more documentation, better organization, complete training, etc.

- [ukk23test01-train-v1.ipynb](ukk23test01-train-v1.ipynb) - generates files containing the NN model that is later used in the RRTMGP-NN model. The implementation uses TensorFlow/Python for training the NN, and Fortran routines are used to generate the training dataset. Based on Ukk22 git main branch.

- [ukk23test01-train-sd-v240823.ipynb](ukk23test01-train-sd-v240823.ipynb) - NN network training for the optical gas radiation problem, running on SDumont.

- [rfmip01-clear-sky.ipynb](rfmip01-clear-sky.ipynb) - runs the RFMIP-CLEAR-SKY example, from ukk23test01 dir, described in [[Ukk22]](references.md#Ukk22). Was "ukk23test01-rfmip-clear-sky.ipynb".

- [ecrad01-NN-gprof.ipynb](ecrad01-NN-gprof.ipynb) - "ecRad-NN". Gprof of ecRad from `ukk23eo01` dir (uses RRTMGP-NN). Was "ukk23eo01-NN-gprof.ipynb".

- [ecrad01-gprof.ipynb](ecrad01-gprof.ipynb) - "ecRad". Gprof of ecrad (without NN) executable from ecrad dir.

- [ecrad-01-sd-v240823.ipynb](ecrad-01-sd-v240823.ipynb) - shows the original ecRad radiation module using conventional numerical method, running on SDumont.

- [ecrad-01-gc.ipynb](ecrad-01-gc.ipynb) - ecRad compiling and running on Google Colab.

## Other files

- [ecrad-radiation-user-guide-2022](ecrad-radiation-user-guide-2022.md) - ecRad Radiation Scheme User Guide document converted from PDF to HTML. Source: [Hogan, R. J. [ecRad radiation scheme: User Guide](https://confluence.ecmwf.int/download/attachments/70945505/ecrad_documentation.pdf?version=5&modificationDate=1655480733414&api=v2). Version 1.5 (June 2022) applicable to ecRad version 1.5.x].

- `*.txt` and `*.yml` are auxiliary files.

- `*.nc` files are of type NetCDF4 and can be browsed and their structure visualized using the Java tool [ToolsUI](https://docs.unidata.ucar.edu/netcdf-java/current/userguide/reading_cdm.html) or using the Python library [netcdf4-python](https://github.com/Unidata/netcdf4-python).

- [env](env.md) :  briefly describes the environment used in the Notebooks.

Please note that due to space constraints on Github, not all files used are present. The [.gitignore](https://github.com/efurlanm/radnn/blob/main/.gitignore) file in the root of the repository contains a list of ignored files. The complete listing of the `src/` directory on the local machine is available in [listingsrc](listingsrc.html).

## Code and data

Due to size restrictions, the data is not present in this repository and must be obtained and installed from several sources:

- <https://zenodo.org/records/7413952> [[Ukk22d]](references.md#Ukk22d)
- <https://zenodo.org/records/5564314> [[Ukk22c]](references.md#Ukk22c)
- <https://zenodo.org/records/7852526> [[Ukk22a]](references.md#Ukk22a)
- <https://zenodo.org/records/7413935> [[Ukk22]](references.md#Ukk22)
- <https://github.com/peterukk/rte-rrtmgp-nn> [[Ukk22]](references.md#Ukk22)
- <https://zenodo.org/records/5833494> [[Ukk21]](references.md#Ukk21)
- <https://zenodo.org/records/4030436> [[Ukk20]](references.md#Ukk20)
- <https://zenodo.org/records/4029138> [[Ukk20]](references.md#Ukk20)

The repositories in <https://github.com/peterukk> contain code and some data, distributed across different repo branches. The RTE+RRTMGP-NN is available on:

- <https://github.com/peterukk/rte-rrtmgp-nn> (there are multiple branches in the repository containing different data files).

- <https://doi.org/10.5281/zenodo.7413935> [[Ukk22]](references.md#Ukk22). The link redirects to: <https://zenodo.org/records/7413935> . "peterukk/rte-rrtmgp-nn: 2.0".

- The Fortran programs and Python scripts used for data generation and model training are found in the directory `examples/rrtmgp-nn-training`.

The training data and the archived version of RTE+RRTMGP-NN 2.0 with its training scripts are available at <https://doi.org/10.5281/zenodo.6576680> (see [[Ukk22d]](references.md#Ukk22d)). The link redirects to: <https://zenodo.org/records/7413952> . "Code and extensive data for training neural networks for radiation, used in "Implementation of a machine-learned gas optics parameterization in the ECMWF Integrated Forecasting System: RRTMGP-NN 2.0" ".

The optimized version of the ecRad radiation scheme integrated with RRTMGP-NN 2.0 can be accessed at: <https://doi.org/10.5281/zenodo.7148329> (see [[Ukk22a]](references.md#Ukk22a)). The link redirects to: <https://zenodo.org/records/7852526> . "Optimized version of the ecRad radiation scheme with new RRTMGP-NN gas optics".

There is a data repository hosted on Zenodo titled "Training and evaluation data for machine learning models emulating the RTE+RRTMGP radiation scheme or its components". In this repository there are 3 versions of the data, each version with a DOI and address:

* Version 3: <https://doi.org/10.5281/zenodo.5833494>
* Version 2: <https://doi.org/10.5281/zenodo.5564314>
* Version 1: <https://doi.org/10.5281/zenodo.5513435>

The complete listing of the `data/` directory is available in [listingdata](listingdata.html).

## Miscellaneous

- [Personal notes](notes.md) - I keep a separate file (page) with random notes.
- [pd1b24](https://github.com/efurlanm/pd1b24) - repository containing code, data, manuscripts and other files used in my research used in the Exams and Qualification Proposal, related to Physics-Informed Machine Learning (PIML) and Physics-Informed Neural Networks (PINN) methods.

<br><sub>Last edited: 2025-05-02 12:05:17</sub>
