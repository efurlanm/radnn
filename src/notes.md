# My personal random notes 

* The Table of Contents, for Markdown and Notebook files, can be viewed in JupyterLab using the "View > Table of Contents" menu item or by pressing Ctrl+Shift+K.

* From <https://github.com/peterukk/rte-rrtmgp-nn/tree/2.0> :
   
      * Instead of the original lookup-table interpolation routine and "eta" parameter to handle the overlapping absorption of gases in a given band, this fork implements neural networks (NNs) to predict optical properties for given atmospheric conditions and gas concentrations, which includes all minor longwave (LW) gases supported by RRTMGP. The NNs predict molecular absorption (LW/SW), scattering (SW) or emission (LW) for all spectral points from an input vector consisting of temperature, pressure and gas concentrations of an atmospheric layer. The models have been trained on 6-7 million samples (LW) spanning a wide range of conditions (pre-industrial, present-day, future...) so that they may be used for both weather and climate applications.

* From <https://zenodo.org/records/7413952> :
   
      * The files contain datasets for training neural network versions of the RRTMGP gas optics scheme (as described in the paper) that are read by `ml_train.py`.
   
      * The ML datasets were generated using the input profiles datasets and running the Fortran programs `rrtmgp_sw_gendata_rfmipstyle.F90` and `rrtmgp_lw_gendata_rfmipstyle.F90` in `rte-rrtmgp-nn/examples/rrtmgp-nn-training`, which call the RRTMGP gas optics scheme.

* From [[Ukk20][Ukk20]] :
   
      * The GPTL profiler was used to profile the code. <https://jmrosinski.github.io/GPTL/>
   
      * Also used to obtain data: RFMIP (Radiative Forcing Model Intercomparison Project).
   
      * RTE+RRTMGP-NN is available on Github. <https://github.com/peterukk/rte-rrtmgp-nn>
   
      * Version archived online: <https://doi.org/10.5281/zenodo.4029138>
        
           * Is supplement to
             <https://github.com/peterukk/rte-rrtmgp-nn/tree/0.9>
   
      * Scripts and data used in this paper are available online: <https://doi.org/10.5281/zenodo.3909653>

* The NN inference and I/O code in RRTGMP-NN is based on Neural-Fortran  [[Cur19][Cur19]].

* **RTE+RRTMGP** [[Pin19][Pin19]] is a recently developed **radiation transfer scheme** for dynamical models combining two codes: Radiative Transfer for Energetics (RTE), which computes fluxes given a description of boundary conditions, source functions and optical properties of the atmosphere, and RRTM for General circulation model applications — Parallel (RRTMGP), which computes optical properties and source functions of the gaseous atmosphere. The **gas optics scheme RRTMGP** uses a k-distribution based on state-of-the-art spectroscopy, and has 256 g-points in the longwave and 224 g-points in the shortwave, which is high compared to many other schemes. [[Ukk20][Ukk20]]
