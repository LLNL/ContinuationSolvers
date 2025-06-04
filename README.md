# HomotopySolver




***

## HomotopySolver
This repository contains two homotopy solvers. First is for constrained optimization via the interior-point method, the second is a mixed nonlinear complementarity problem solver.

## Description
With this project one can solve nonlinear complementarity problems as defined by a GeneralNLMCProblem (see problems/NLMCProblems.hpp). One can also solve constrained optimization problems by the interior-point method as defined by a ParGeneralOptProblem (see problems/OptProblems.hpp).

## Installation
1. Install [mfem](https://mfem.org/). 
2. Edit the MFEM_DIR and MFEM_BUILD_DIR variables in homotopysolver/Makefile and problems/Makefile
3. Make and run e.g., (from homotopysolver/problems)
make TestProblem1
mpirun -np 4 ./TestProblem1   

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tucker Hartland (hartland1@llnl.gov).

## Authors and acknowledgment
This code is developed by Tucker Hartland. For details on the Newton homotopy method see "A filter trust-region Newton continuation method for nonlinear complementarity problems", Cosmin G. Petra, Nai-Yuan Chiang, Jingyi "Frank" Wang, Tucker Hartland, Eric Chin, and Mike Puso (submitted). LLNL Release #: LLNL-JRNL-869761

This code has been developed with support from the LLNL LDRD project 23-ERD-017 (PI: Cosmin G. Petra).

## Copyright
Copyright (c) 2025-2025, Lawrence Livermore National Security, LLC. All rights reserved. Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-2006107. HomotopySolver is free software; you can modify it and/or redistribute it under the terms of the BSD-3 clause license. See COPYRIGHT AND LICENSE for complete copyright and license information.
