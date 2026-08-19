# NPP

MATLAB code for *The Normal Procrustes Problem: A Riemannian Optimization Approach*. 
Minimizes ||AX - Y||_F^2 over normal A by reducing to an optimization
over U(m) or O(m) and running Manopt's trust-region solver. Taking X = I_m gives the
Closest Normal Matrix Problem.

Requires MATLAB and [Manopt](https://www.manopt.org).

`MATLAB-Files/` holds the scripts; each takes no arguments, prints the table numbers, and
writes its figures as PNGs. `Figures/Paper/` are the ones in the paper,
`Figures/Supplementary/` the rest (mostly gradient-norm trajectories and the remaining
(m,n) cells). `diary/` has the console output from the reported runs. Everything seeds
with `rng(0)`.
