This repository includes code for simulating the Gray-Scott reaction diffusion system and a variety of extensions thereof.

In the most basic version, the fluid is static and the Gray-Scott patterns emerge and evolve to steady state. However, the Rayleigh number can be increased such that stronger (eventually time-varying and turbulent) convection patterns emerge, and the reaction enthalpies and temperature dependencies of the reactions can be increased such that the GS system interacts strongly with the convection. 

To compile use the following (or similar):
mpicc -O3 -Wall -Wextra -std=c99 -ffp-contract=off -o reac_diff_5a reac_diff_5a.c -lm

To run use the following (adjust number of processes as needed):
mpirun -n 4 ./reac_diff_5a

Also attached is a simple Python script for making an animation of the model run. Use the following command to run that:
python anim.py opts_reac_1.txt
