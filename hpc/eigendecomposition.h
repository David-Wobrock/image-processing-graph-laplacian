#include <petscmat.h>

void EigendecompositionLargest(Mat A, const unsigned int num_eigenpairs, Mat* eigenvectors, Mat* eigenvalues, Mat* eigenvalues_inv);
void EigendecompositionSmallest(Mat A, const unsigned int num_eigenpairs, Mat* eigenvectors, Mat* eigenvalues, Mat* eigenvalues_inv);
