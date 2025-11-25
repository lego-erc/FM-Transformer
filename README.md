# Structural Overview

- ### cfm: Main Continuous Flow Matching Components. 

    - cfm_trafo_x.py: xTransformer construction and Input processing. 
    - rie_int.py: Utilities for integrating the ODE on the manifolds.  

- ### data

    - dataloaders.py: gets data from a file and preprocesses (e.g. cutoff energy, filters nan's)

- ###  discrete: First ideas for discrete modelling/tokenization.

- ### geometry: Various utilities for geometry transformations.

    - energy_proj.py: normalization for the magnitute of the momentum. 
    - gen_base.py: flow base sample generation. 
    - path_sample_mult.py: enables projections and path computations on products of manifolds. 
    - raytracing_proj.py: projects to the other side of the cube, can add noise. 
    - vmf_sampling.py: sampling uniformly on a sphere, transformations from cartesian to spherical.
    
- ### multiplicity

    - gen_mult.py: model to generate a distribution for the number of outgoing particles.

- ### testing: some test notebooks and lightning scripts. 

- ### utils

    - generate.py: generate outgoing particles from a batch of incoming and model checkpoints for flow and multiplicity.
    - wrappers.py: projection wrapper for manifold and normalization, as well as the lighning model wrapper