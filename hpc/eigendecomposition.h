#include <petscmat.h>

void EigendecompositionLargest(Mat A, const PetscInt num_eigenpairs, Mat* eigenvectors, Mat* eigenvalues, Mat* eigenvalues_inv);
void EigendecompositionSmallest(Mat A, const PetscInt num_eigenpairs, Mat* eigenvectors, Mat* eigenvalues, Mat* eigenvalues_inv);
