# Some references and comments

The references below contain additional information such as comments, code, and data. The GitHub repositories cited use the GIT version control system, and two terms appear: "Branches" and "Tags". Branches are used for parallel development and can change as new commits are added. Tags are used to mark important points in the project's history and are generally immutable. Zenodo repositories, in addition to files, have descriptions with more information, and even more references.

## Papers

(unordered)

**[Ukk24]**<a name="Ukk24" href="#Ukk24"></a>: 
P. Ukkonen and R. J. Hogan, “Twelve Times Faster yet Accurate: A New State-Of-The-Art in Radiation Schemes via Performance and Spectral Optimization,” Journal of Advances in Modeling Earth Systems, vol. 16, no. 1, 2024, doi: 10.1029/2023MS003932.

- Ⓖ<https://github.com/ecmwf-ifs/ecrad>

**[Ukk23]**<a name="Ukk23" href="#Ukk23"></a>: 
Ukkonen, P., & Hogan, R. J. (2023). Implementation of a machine-learned gas optics parameterization in the ECMWF Integrated Forecasting System: RRTMGP-NN 2.0. Geoscientific Model Development, 16(11), 3241–3261. <https://gmd.copernicus.org/articles/16/3241/2023/> (redirected from <https://doi.org/10.5194/gmd-16-3241-2023>).

- RTE+RRTMGP-NN scheme: 
  Ⓐ<https://github.com/peterukk/rte-rrtmgp-nn> (git branch **main**).

- The Fortran and Python code used for data generation and model training comes from [[Ukk22](#Ukk22)] <https://doi.org/10.5281/zenodo.7413935> . They are in the subdirectory `examples/rrtmgp-nn-training` .

- Training data and archived version of RTE+RRTMGP-NN 2.0 with its training scripts: <https://zenodo.org/records/7413952> (redirected from <https://doi.org/10.5281/zenodo.6576680>).

- The optimized version of the ecRad radiation scheme integrated with RRTMGP-NN 2.0 comes from [[Ukk22a](#Ukk22a)] <https://zenodo.org/records/7852526> (redirected from <https://doi.org/10.5281/zenodo.7148329>).

- The RRTMGP-NN scheme is described in [[Ukk20](#Ukk20)] .

<!-- -------------------------------------- -->

**[Ukk22c]**<a name="Ukk22c" href="#Ukk22c"></a>: 
Ukkonen, P. (2022). Exploring Pathways to More Accurate Machine Learning Emulation of Atmospheric Radiative Transfer. Journal of Advances in Modeling Earth Systems, *14*(4), e2021MS002875. <https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021MS002875> (redirected from <https://doi.org/10.1029/2021MS002875>).

- Excerpt from the paper: "The code used in this work is available on Github on a dedicated branch of the RTE+RRTMGP-NN package (Ⓒ<https://github.com/peterukk/rte-rrtmgp-nn/tree/nn_dev/examples/emulator-training>) (git branch **nn_dev**); which includes Python and Fortran code for data retrieval, pre-processing, data generation, model training, and model evaluation. The machine learning input-output data can be found on Zenodo (②<https://doi.org/10.5281/zenodo.5564314>)." (see [[Ukk21](#Ukk21)])
  
  - "5564314" is version 2 of the files. There is also version 3: "5833494" (③<https://doi.org/10.5281/zenodo.5833494>). (see [[Ukk21](#Ukk21)])

<!-- -------------------------------------- -->

**[Ukk20]**<a name="Ukk20" href="#Ukk20"></a>: 
Ukkonen, P., Pincus, R., Hogan, R. J., Pagh Nielsen, K., & Kaas, E. (2020). Accelerating Radiation Computations for Dynamical Models With Targeted Machine Learning and Code Optimization. Journal of Advances in Modeling Earth Systems, 12(12). <https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020MS002226> (redirected from <https://doi.org/10.1029/2020MS002226>).

- Supplementary Python code and data used to train NN. <https://zenodo.org/records/4030436>.

- RRTMGP-NN 2020 code and data. <https://zenodo.org/records/4029138>. Is supplement to Ⓓ<https://github.com/peterukk/rte-rrtmgp-nn/tree/0.9> .

<!-- -------------------------------------- -->

**[Cur19]**<a name="Cur19" href="#Cur19"></a>: 
Curcic, M. (2019). A parallel Fortran framework for neural networks and deep learning. *ACM SIGPLAN Fortran Forum*, *38*(1), 4–21. <https://dl.acm.org/doi/10.1145/3323057.3323059> (redirected from <https://doi.org/10.1145/3323057.3323059>).

<!-- -------------------------------------- -->

**[Pin19]**<a name="Pin19" href="#Pin19"></a>: 
Pincus, R., Mlawer, E. J., & Delamere, J. S. (2019). Balancing Accuracy, Efficiency, and Flexibility in Radiation Calculations for Dynamical Models. Journal of Advances in Modeling Earth Systems, *11*(10), 3074–3089. <https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019MS001621> (redirected from <https://doi.org/10.1029/2019MS001621>).

- Includes Neural-Fortran.

- RTE+RRTMGP is a set of codes for computing radiative fluxes in planetary atmospheres. <https://github.com/RobertPincus/rte-rrtmgp> .

<!-- -------------------------------------- -->

## Repositories

**[Ukk22]**<a name="Ukk22" href="#Ukk22"></a> [Repository]:
Ukkonen, P., et al. (Dec 8, 2022). "peterukk/rte-rrtmgp-nn: 2.0" [Code]. (includes NN training). <https://zenodo.org/records/7413935> (redirected from <https://doi.org/10.5281/zenodo.7413935> ). (repository mentioned in the paper [[Ukk23](#Ukk23)]).

- Code and data repository. Description: "*Dec 2022: "Official 2.0 release" corresponding to submitted GMD (previously a JAMES preprint earlier in the year) article describing RRTMGP-NN implementation in ecRad and prognostic testing in the IFS*".

- It is a supplement to another repository: Ⓔ<https://github.com/peterukk/rte-rrtmgp-nn/tree/2.0> (git branch **main** tag **2.0**).

<!-- -------------------------------------- -->

**[Ukk22a]**<a name="Ukk22a" href="#Ukk22a"></a> [Repository]:
Ukkonen, P. (Oct 5, 2022). Optimized version of the ecRad radiation scheme with new RRTMGP-NN gas optics. (code and data). (does not include the NN training). The development version of ecRad 1.6 which includes CPU performance optimizations and ecCKD (already included in official version 1.5), RRTMGP and RRTMGP-NN gas optics schemes. <https://zenodo.org/records/7852526> (redirected from <https://doi.org/10.5281/zenodo.7852526>).

- From the repo: " (...) the is most up-to-date optimized ecRad code, see the **clean_no_opt_testing** branch in this github repository:" Ⓕ<https://github.com/peterukk/ecrad-opt/tree/clean_no_opt_testing> .

- The code is used in the papers
  
  - Ⓗ<https://gmd.copernicus.org/articles/16/3241/2023/>
  - Ⓘ<https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021MS002875>

<!-- -------------------------------------- -->

**[Ukk22d]**<a name="Ukk22d" href="#Ukk22d"></a> [Repository]:
Ukkonen, P. (May 25, 2022). Code and extensive data for training neural networks for radiation, used in "Implementation of a machine-learned gas optics parameterization in the ECMWF Integrated Forecasting System: RRTMGP-NN 2.0". <https://zenodo.org/records/7413952> (redirected from <https://doi.org/10.5281/zenodo.7413952>).

- The description contained in the repository cites the references
  
  - Ⓐ<https://github.com/peterukk/rte-rrtmgp-nn>
  - Ⓑ<https://github.com/peterukk/rte-rrtmgp-nn/tree/main/examples/rrtmgp-nn-training>

<!-- -------------------------------------- -->

**[Ukk21]**<a name="Ukk21" href="#Ukk21"></a> [Repository]:
Ukkonen, P. (2021). Training and evaluation data for machine learning models emulating the RTE+RRTMGP radiation scheme or its components. (repository cited in the paper [[Ukk22c](#Ukk22c)])

- Versions
  
  - ③<https://doi.org/10.5281/zenodo.5833494>
  - ②<https://doi.org/10.5281/zenodo.5564314>
  - ①<https://doi.org/10.5281/zenodo.5513435>

- Excerpt from the repository description: "The Fortran program and instructions (...) can be found at Ⓒ<https://github.com/peterukk/rte-rrtmgp-nn/tree/nn_dev/examples/emulator-training>".

<!-- -------------------------------------- -->

## Main links summary

- Ⓐ<https://github.com/peterukk/rte-rrtmgp-nn>
- Ⓑ<https://github.com/peterukk/rte-rrtmgp-nn/tree/main/examples/rrtmgp-nn-training>
- Ⓒ<https://github.com/peterukk/rte-rrtmgp-nn/tree/nn_dev/examples/emulator-training>
- Ⓓ<https://github.com/peterukk/rte-rrtmgp-nn/tree/0.9>
- Ⓔ<https://github.com/peterukk/rte-rrtmgp-nn/tree/2.0>
- Ⓕ<https://github.com/peterukk/ecrad-opt/tree/clean_no_opt_testing>
- Ⓖ<https://github.com/ecmwf-ifs/ecrad>
- Ⓗ<https://gmd.copernicus.org/articles/16/3241/2023/>
- Ⓘ<https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021MS002875>

## Other

[Other references](references.bib) - file in BibTex format containing  other references and also the previous references on this page.

<br><sub>Last edited: 2025-02-22 21:16:58</sub>
