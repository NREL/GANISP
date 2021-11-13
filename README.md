# GANISP: a GAN-assisted Importance SPlitting Probability estimator


## Requirements
- Python v3.7.6
- tensorflow v2.3.0
- numpy v1.19.1
- matplotlib v3.2.2
- mpi4py v3.0.3
- openMPI v4.0.4


## Purpose

Designing manufacturing processes with high yield and strong reliability relies on effective methods for rare event estimation.
Genealogical importance splitting reduces the variance of rare event probability estimators by iteratively selecting and replicating realizations that are headed towards a rare event. The replication step is difficult when applied to deterministic systems where the initial conditions of the offspring realizations need to be modified. Typically, a random perturbation is applied to the offspring to differentiate their trajectory from the parent realization. However, this random perturbation strategy may be effective for some systems while failing for others, preventing variance reduction in the probability estimate. This work seeks to address this limitation using a generative model such as a Generative Adversarial Network (GAN) to generate perturbations that are consistent with the attractor of the dynamical system.

## Code description

`BruteForce/` : module that run multiple realizations of the Kuramoto-Sivashinsky equation (KSE) and the Lorenz 96 (L96) equation and compute the CDF of a QoI. MPI-parallelization across the realizations is available.

`Generative/`: module in charge of preparing the data and train a GAN which will ultimately be used for cloning realizations. The GAN is conditioned on the reaction coordinate used to track the realizations. To encourage diversity in the generated samples, the moments of the conditional distribution to sample are computed a priori. Computation of conditional moments and GAN training implementation heavily draw from [Diverse Super-resolution (SR) repository](https://github.com/NREL/diversity_SR/tree/master/diversity_SR). Available for the KSE.

`ISP_parallel/`: module that implements the baseline implementation of the genealogical importance splitting method (with random cloning). Available for the KSE and L96.

`GANISP_parallel/`: module that implements the GANISP, with a GAN-assisted cloning method. Available for the KSE. 


## Contact

Malik Hassanaly: (malik.hassanaly!at!nrel!gov)

## Acknowledgements


This work was authored by the National Renewable Energy Laboratory (NREL), operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. This work was supported by the Laboratory Directed Research and Development (LDRD) Program at NREL. The research was performed using computational resources sponsored by the Department of Energy's Office of Energy Efficiency and Renewable Energy and located at the National Renewable Energy Laboratory. The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government. The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes.



