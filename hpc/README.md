# Image Processing using Graph Laplacian Operator - HPC

Implemented in C using [PETSc](http://www.mcs.anl.gov/petsc/).

Installation requires:

* [PETSc](http://www.mcs.anl.gov/petsc/) (which requires MPI, BLAS, LaPACK)
* [SLEPc](http://slepc.upv.es/)
* [Elemental](http://libelemental.org/) (which requires METIS and parMETIS)

Compile:

`make`

The program has 3 modes:

* Approximating the Laplacian eigenvalues using the self-implemented inverse power method (solving systems of linears equations)
* Approximating the eigenvalues using SLEPc
* Computing the entire matrices, no approximation (very memory consuming)

Options when running:

* `-f FILENAME` - input image. Only supports **grayscale PNG** images
* `-num_eigvals NUM` - number of eigenvalues to compute. Defaults to `sample_size - 1`
* `-no_approx` - compute full matrices, no approximation, no eigenvalue computation
* `-use_slepc` - compute the eigenvalues using SLEPc
* `-opti_gs` - when using the inverse iteration, only apply Gram-Schmidt every other outer iteration
* `-inv_it_epsilon EPS` - set the epsilon of the outer iteration, the value the residual norm must reach
