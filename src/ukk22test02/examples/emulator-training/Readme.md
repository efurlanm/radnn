**Generate training data for ML emulators** 

Code for generating training data to replace RRTMGP and/or RTE, train emulators, and evaluating these emulators

These example Fortran programs load atmospheric profiles from a netCDF file, compute gas optical properties with RRTMGP, optionally compute cloud optical properties, and then compute shortwave fluxes with RTE. They can also be used to generate training data (allsky_sw_gendata.F90), or test an emulator by replacing a physical component with it (allsky_sw_testmodels.F90) - see below for examples. In the `scripts/` subdirectory, Python code for training neural network-based emulators can be found, as well as code for retrieving and pre-processing CAMS reanalysis data.

This code is associated with a paper submitted to Journal of Advances in Modeling Earth Systems (JAMES) and all data has been uploaded to [Zenodo](https://doi.org/10.5281/zenodo.5513434)

------

The use and compilation of RTE+RRTMGP-NN is similar to the original code, but a BLAS library is required. If you're not using ifort+MKL then [BLIS](https://github.com/flame/blis) is recommended

1. Build the RTE+RRTMGP libraries in `../../build/`. This will require setting
   environment variables `FC` for the Fortran compiler and `FCFLAGS`, or creating
   `../../build/Makefile.conf` with that information (see Makefile.conf.X for examples). Other optional variables include:
- (Optional) Set `GPTL=1` to use the GPTL timing library, or `GPTL=2` to use GPTL with PAPI performance counters in order to measure computational intensity. You also need to provide location `TIME_DIR`. If GPTL was built with OpenMP then you will need to add -fopenmp to compilation flags which can be done with `USE_OPENMP=1`
- (Optional) Set `USE_OPENACC=1` if you want to use OpenACC+CUDA for GPU acceleration (see Makefile.conf.nvfortran for example compilation flags)
- (Optional) Single precision is enabled by default, to use double prec. set `DOUBLE_PRECISION=1`
2. Build the executables in this directory, which will first require setting the folowing variables in the environment or via file Makefile.libs:
- (Required) The locations of the netCDF C and Fortran libraries and module files `NCHOME` and `NFHOME`
- (Required) Specify the BLAS Library (e.g. BLIS) e.g.`BLASLIB=blis` and its location `BLAS_DIR`. If you are using Intel MKL then set `BLASLIB=mkl` and ensure `MKLROOT` is specified instead. 
3. After compiling the allsky_sw_gendata and allsky_sw_testmodels, they are used in the following way.

Compute shortwave fluxes using refererence code and save them to fluxes subdirectory:

` ./allsky_sw_testmodels 4 data_input/CAMS_2015_RFMIPstyle.nc ../../rrtmgp/data/rrtmgp-data-sw-g224-2018-12-04.nc ../../extensions/cloud_optics/rrtmgp-cloud-optics-coeffs-sw.nc fluxes/CAMS_2015_rsud_REFERENCE.nc`

Save input output data for training machine learning emulators (modify Fortran program to specify the outputs and other options):

` ./allsky_sw_gendata 8 data_input/CAMS_2015_RFMIPstyle.nc ../../rrtmgp/data/rrtmgp-data-sw-g224-2018-12-04.nc ../../extensions/cloud_optics/rrtmgp-cloud-optics-coeffs-sw.nc  /media/peter/samsung/data/CAMS/ml_training/RADSCHEME_data_g224_CAMS_2014.nc `

Test an existing ML emulator, e.g. neural network to accelerate RRTMGP kernel (saving fluxes):

`/allsky_sw_testmodels 4 data_input/CAMS_2015_RFMIPstyle.nc ../../rrtmgp/data/rrtmgp-data-sw-g224-2018-12-04.nc ../../extensions/cloud_optics/rrtmgp-cloud-optics-coeffs-sw.nc fluxes/tmp.nc rrtmgp ../../neural/data/tau-sw-abs-7-16-16-CAMS-NEW.txt ../../neural/data/tau-sw-ray-7-16-16-CAMS-NEW-mae.txt ` 

Or emulate the entire scheme using the final RADSCHEME model:

`./allsky_sw_testmodels 5120 data_input/CAMS_2015_RFMIPstyle.nc ../../rrtmgp/data/rrtmgp-data-sw-g224-2018-12-04.nc ../../extensions/cloud_optics/rrtmgp-cloud-optics-coeffs-sw.nc fluxes/tmp.nc both ../../neural/data/radscheme-128-128-128-hybridloss_new.txt ` 

Or emulate reflectance-transmittance computations using the final REFTRANS model:

` ./allsky_sw_testmodels 4 data_input/CAMS_2015_RFMIPstyle.nc ../../rrtmgp/data/rrtmgp-data-sw-g224-2018-12-04.nc ../../extensions/cloud_optics/rrtmgp-cloud-optics-coeffs-sw.nc fluxes/tmp.nc rte-reftrans ../../neural/data/reftrans-12-12-sinemse2-mae-NEW.txt` 

To not include clouds, provide "none" in place of the cloud coefficient file. 
