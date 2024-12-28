# References

The following references have additional information such as comments, code, and data.


## References to code and data

(unordered)

<a name="Ukk23"></a>**[Ukk23]** Ukkonen, P., & Hogan, R. J. (2023). Implementation of a machine-learned gas optics parameterization in the ECMWF Integrated Forecasting System: RRTMGP-NN 2.0. Geoscientific Model Development, 16(11), 3241–3261. <https://doi.org/10.5194/gmd-16-3241-2023> (redirects to <https://gmd.copernicus.org/papers/16/3241/2023>).

- RTE+RRTMGP-NN scheme: <https://github.com/peterukk/rte-rrtmgp-nn> .
- The Fortran and Python code used for data generation and model training comes from [[Ukk22](#Ukk22)] <https://doi.org/10.5281/zenodo.7413935> . They are in the subdirectory `examples/rrtmgp-nn-training` .
- Training data and archived version of RTE+RRTMGP-NN 2.0 with its training scripts: <https://zenodo.org/records/7413952> (redirected from <https://doi.org/10.5281/zenodo.6576680>).
- The optimized version of the ecRad radiation scheme integrated with RRTMGP-NN 2.0 comes from [[Ukk22a](#Ukk22a)] <https://zenodo.org/records/7852526> (redirected from <https://doi.org/10.5281/zenodo.7148329>).
- The RRTMGP-NN scheme is described in [[Ukk20](#Ukk20)] .


<a name="Ukk22"></a>**[Ukk22]** Ukkonen, P., et al. (Dec 8, 2022). "peterukk/rte-rrtmgp-nn: 2.0" (Code). (see also [[Ukk23](#Ukk23)]). (includes NN training). Code and data repository. Description: "*Dec 2022: "Official 2.0 release" corresponding to submitted GMD (previously a JAMES preprint earlier in the year) article describing RRTMGP-NN implementation in ecRad and prognostic testing in the IFS*". <https://doi.org/10.5281/zenodo.7413935> (redirects to <https://zenodo.org/records/7413935>).

- It's probably a mirror: <https://github.com/peterukk/rte-rrtmgp-nn/tree/2.0> .


<a name="Ukk22d"></a>**[Ukk22d]** Ukkonen, P. (2022). Code and extensive data for training neural networks for radiation, used in "Implementation of a machine-learned gas optics parameterization in the ECMWF Integrated Forecasting System: RRTMGP-NN 2.0" [Dataset]. <https://doi.org/10.5281/zenodo.7413952> (redirects to <https://zenodo.org/records/7413952>).


<a name="Ukk22a"></a>**[Ukk22a]** Ukkonen, P. (Oct 5, 2022). Optimized version of the ecRad radiation scheme with new RRTMGP-NN gas optics. (code and data) (does not include the NN training).

- The development version of ecRad 1.6 which includes CPU performance optimizations and ecCKD (already included in official version 1.5), RRTMGP and RRTMGP-NN gas optics schemes. <https://zenodo.org/records/7852526> (redirects from <https://doi.org/10.5281/zenodo.7852526>). There is most up-to-date optimized ecRad code, see the clean_no_opt_testing branch in this github repository: <https://github.com/peterukk/ecrad-opt> . 

- Code: <https://github.com/peterukk/ecrad-opt/tree/clean_no_opt_testing> .


<a name="Ukk22c"></a>**[Ukk22c]** Ukkonen, P. (2022). Exploring Pathways to More Accurate Machine Learning Emulation of Atmospheric Radiative Transfer. Journal of Advances in Modeling Earth Systems, *14*(4), e2021MS002875. <https://doi.org/10.1029/2021MS002875> (redirects to: <https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021MS002875>).


<a name="Ukk21"></a>**[Ukk21]** Ukkonen, P. (2021). Training and evaluation data for machine learning models emulating the RTE+RRTMGP radiation scheme or its components.

- Repository version 3: <https://doi.org/10.5281/zenodo.5833494>
- Repository version 2: <https://doi.org/10.5281/zenodo.5564314>
- Repository version 1: <https://doi.org/10.5281/zenodo.5513435>


<a name="Ukk20"></a>**[Ukk20]** Ukkonen, P., Pincus, R., Hogan, R. J., Pagh Nielsen, K., & Kaas, E. (2020). Accelerating Radiation Computations for Dynamical Models With Targeted Machine Learning and Code Optimization. Journal of Advances in Modeling Earth Systems, 12(12). <https://doi.org/10.1029/2020MS002226> (redirects to <https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020MS002226>).

- Supplementary Python code and data used to train NN. <https://zenodo.org/records/4030436>.
- RRTMGP-NN 2020 code and data. <https://zenodo.org/records/4029138>. Is supplement to <https://github.com/peterukk/rte-rrtmgp-nn/tree/0.9> .


<a name="Cur19"></a>**[Cur19]** Curcic, M. (2019). A parallel Fortran framework for neural networks and deep learning. *ACM SIGPLAN Fortran Forum*, *38*(1), 4–21. <https://doi.org/10.1145/3323057.3323059> (redirects to <https://dl.acm.org/doi/10.1145/3323057.3323059>).


<a name="Pin19"></a>**[Pin19]** Pincus, R., Mlawer, E. J., & Delamere, J. S. (2019). Balancing Accuracy, Efficiency, and Flexibility in Radiation Calculations for Dynamical Models. Journal of Advances in Modeling Earth Systems, *11*(10), 3074–3089. <https://doi.org/10.1029/2019MS001621> (redirects to <https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019MS001621>).

- includes Neural-Fortran
- RTE+RRTMGP is a set of codes for computing radiative fluxes in planetary atmospheres. <https://github.com/RobertPincus/rte-rrtmgp> .


## Other references

[Other references](references.bib) - file in BibTex format containing  other references and also the previous references on this page.

