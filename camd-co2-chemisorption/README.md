# Computer-Aided Molecular Design (CAMD) for CO₂ Chemisorption Solvents

## Overview
This project implements a Computer-Aided Molecular Design (CAMD) framework in Python using the Pyomo toolkit to identify industrially feasible novel solvents for CO₂ capture via chemisorption. The optimisation problem is formulated using Pyomo as a mixed-integer nonlinear programming (MINLP) model.

The objective is to design molecules that balance:
- High CO₂ absorption capacity
- Fast reaction kinetics
- Low regeneration energy
- Industrial feasibility (e.g. stability, safety)

## Methodology
The approach combines:
- Group contribution methods for thermodynamic property estimation
- Reaction-based modelling of chemisorption
- Optimisation of molecular structures under process constraints using **Pyomo**

Candidate molecules are evaluated against key performance indicators relevant to post-combustion carbon capture.

## Model Description
The model:
- Defines molecular building blocks (functional groups)
- Predicts properties such as:
  - Heat of absorption
  - Solvent capacity
  - Thermodynamic stability
- Applies constraints to ensure practical feasibility

The optimisation framework searches for optimal solvent designs.

## File Structure
- `camd_model.py` — implementation of CAMD framework and optimisation

## Key Features
- Systematic solvent design via molecular structure selection
- Incorporation of industrial constraints
- Quantitative evaluation of CO₂ capture performance

## Scientific Context
Carbon capture is a key technology in decarbonisation. 
Traditional solvents (e.g. MEA) are effective but suffer from:
- High energy penalties
- Degradation issues

This work explores next-generation solvents using computational design methods.

## Dependencies
- NumPy
- Pyomo
- Compatible optimistation solver:
    - BARON (for global MINLP)
    - Alt: IPOPT, GLPK etc.
- SciPy (optional for graphing)
  
## How to Run
```bash
pip install numpy pyomo
# pip install scipy
python camd_model.py
