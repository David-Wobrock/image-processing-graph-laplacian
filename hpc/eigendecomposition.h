#include <petscmat.h>

void Eigendecomposition(Mat A, const unsigned int num_eigenpairs, Mat* eigenvectors, Mat* eigenvalues, Mat* eigenvalues_inv, Mat* eigenvalues_inv_sqrt);
void EigendecompositionAndOrthogonalisation(Mat A, Mat B, Mat* phi, Mat* Pi);
