# Notes

This page contains some of my random personal notes, which I make over time.



- **From** <https://github.com/peterukk/rte-rrtmgp-nn/tree/2.0> 
  
  - Instead of the original lookup-table interpolation routine and "eta" parameter to handle the overlapping absorption of gases in a given band, this fork implements neural networks (NNs) to predict optical properties for given atmospheric conditions and gas concentrations, which includes all minor longwave (LW) gases supported by RRTMGP. The NNs predict molecular absorption (LW/SW), scattering (SW) or emission (LW) for all spectral points from an input vector consisting of temperature, pressure and gas concentrations of an atmospheric layer. The models have been trained on 6-7 million samples (LW) spanning a wide range of conditions (pre-industrial, present-day, future...) so that they may be used for both weather and climate applications.

- **From** <https://zenodo.org/records/7413952>
  
  - The files contain datasets for training neural network versions of the RRTMGP gas optics scheme (as described in the paper) that are read by `ml_train.py`.
  
  - The ML datasets were generated using the input profiles datasets and running the Fortran programs `rrtmgp_sw_gendata_rfmipstyle.F90` and `rrtmgp_lw_gendata_rfmipstyle.F90` in `rte-rrtmgp-nn/examples/rrtmgp-nn-training`, which call the RRTMGP gas optics scheme.

- **From** [[Ukk20](references.md#Ukk20)]
  
  - The GPTL profiler was used to profile the code. <https://jmrosinski.github.io/GPTL/>
  - Also used to obtain data: RFMIP (Radiative Forcing Model Intercomparison Project).
  - RTE+RRTMGP-NN is available on Github. <https://github.com/peterukk/rte-rrtmgp-nn>
  - Version archived online: <https://doi.org/10.5281/zenodo.4029138>
    - Is supplement to
      
            <https://github.com/peterukk/rte-rrtmgp-nn/tree/0.9>
  - Scripts and data used in this paper are available online: <https://doi.org/10.5281/zenodo.3909653>
  - "The NNs, implemented in Fortran and utilizing BLAS for batched inference, are faster by a factor of 1–6, depending on the software and hardware platforms. We combined the accelerated gas optics with a refactored radiative transfer solver, resulting in clear-sky longwave (shortwave) fluxes being 3.5 (1.8) faster to compute on an Intel platform."
  - "radiation is often one of the most expensive components in climate models, accounting for nearly 50% of the runtime of the ECHAM atmospheric model in coarse-resolution configurations (Cotronei & Slawig, 2020)"

- **RTE+RRTMGP** [[Pin19](references.md#Pin19)] 
  
  - is a recently developed **radiation transfer scheme** for dynamical models combining two codes: Radiative Transfer for Energetics (RTE), which computes fluxes given a description of boundary conditions, source functions and optical properties of the atmosphere, and RRTM for General circulation model applications — Parallel (RRTMGP), which computes optical properties and source functions of the gaseous atmosphere. The **gas optics scheme RRTMGP** uses a k-distribution based on state-of-the-art spectroscopy, and has 256 g-points in the longwave and 224 g-points in the shortwave, which is high compared to many other schemes. [[Ukk20](references.md#Ukk20)]

- **Neural Fortran** 
  
  - The NN inference and I/O code in RRTGMP-NN is based on Neural-Fortran  [[Cur19](references.md#Cur19)].

## Documentation

The documentation uses the [MkDocs](https://www.mkdocs.org/) static documentation page generator and the Markdown files available in the subdirectories. To write the documentation on the local machine, the `mkdocs serve` command for the "write-build-check-repeat-loop" is used, and the final documentation is generated using `mkdocs build`.

## Conda & JupyterLab

### Install

```bash
$ wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
$ bash Miniforge3.sh -b -p "${HOME}/conda"
$ rm Miniforge3.sh
$ source $HOME/conda/bin/activate
$ conda install -y jupyterlab jupyterlab_code_formatter black yapf
$ conda deactivate
```

### Run

```bash
$ source $HOME/conda/bin/activate
$ jupyter-lab --notebook-dir=$HOME --no-browser --NotebookApp.token='' --ip=0.0.0.0 --port=8888
```

### JupyterLab Table of Contents (TOC)

* [JupyterLab](https://jupyter.org/) (JL) was used for most of the files in the repository, both for viewing and working on them. The TOC for Markdown and Notebook files can be viewed in JL using the "View > Table of Contents" menu item or by pressing Ctrl+Shift+K.

<br><sub>Last edited: 2025-05-01 13:50:40</sub>
