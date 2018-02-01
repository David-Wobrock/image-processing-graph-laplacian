#include "eigendecomposition.h"
#include <petscvec.h>
#include <slepceps.h>
#include <math.h>

/* Assuming that A is symmetric
Pass uninitialised eigenvectors, eigenvalues and eigenvalues_inv.
Any of these 4 can be NULL if the output is not wanted.
eigenvalues and eigenvalues_inv will be filled on the diagonal and be setted as MPIAIJ.
eigenvectors will be of type MPIDENSE
*/
static void Eigendecomposition(Mat A, const unsigned int num_eigenpairs, Mat* eigenvectors, Mat* eigenvalues, Mat* eigenvalues_inv, EPSWhich which)
{
    EPS eps;
    EPSCreate(PETSC_COMM_WORLD, &eps);
    EPSSetOperators(eps, A, NULL);
    EPSSetProblemType(eps, EPS_HEP); // Symmetric
    EPSSetDimensions(eps, num_eigenpairs, PETSC_DEFAULT, PETSC_DEFAULT);
    EPSSetType(eps, EPSKRYLOVSCHUR);
    EPSSetWhichEigenpairs(eps, which);
    EPSSetFromOptions(eps);

    EPSSolve(eps);

    // Put eigenvalues and eigenvectors into Mat structure
    PetscScalar* values;
    PetscInt* row_indices, sample_size;
    PetscScalar eigval, eigval_inv;
    MatGetSize(A, &sample_size, NULL);
    Vec eigvec; // To store the retrieved eigenvector while iterating
    VecCreate(PETSC_COMM_WORLD, &eigvec);
    VecSetSizes(eigvec, PETSC_DECIDE, sample_size);
    VecSetFromOptions(eigvec);
    VecAssemblyBegin(eigvec);
    VecAssemblyEnd(eigvec);

    // Creating needed matrices
    if (eigenvectors)
    {
        MatCreate(PETSC_COMM_WORLD, eigenvectors);
        MatSetSizes(*eigenvectors, PETSC_DECIDE, PETSC_DECIDE, sample_size, num_eigenpairs);
        MatSetType(*eigenvectors, MATMPIDENSE);
        MatSetFromOptions(*eigenvectors);
        MatSetUp(*eigenvectors);
    }
    if (eigenvalues)
    {
        MatCreate(PETSC_COMM_WORLD, eigenvalues);
        MatSetSizes(*eigenvalues, PETSC_DECIDE, PETSC_DECIDE, num_eigenpairs, num_eigenpairs);
        MatSetType(*eigenvalues, MATMPIAIJ);
        MatSetFromOptions(*eigenvalues);
        MatSetUp(*eigenvalues);
    }
    if (eigenvalues_inv)
    {
        MatCreate(PETSC_COMM_WORLD, eigenvalues_inv);
        MatSetSizes(*eigenvalues_inv, PETSC_DECIDE, PETSC_DECIDE, num_eigenpairs, num_eigenpairs);
        MatSetType(*eigenvalues_inv, MATMPIAIJ);
        MatSetFromOptions(*eigenvalues_inv);
        MatSetUp(*eigenvalues_inv);
    }

    PetscInt istart, iend;
    VecGetOwnershipRange(eigvec, &istart, &iend);
    values = (PetscScalar*) malloc(sizeof(PetscScalar) * (iend-istart));
    row_indices = (PetscInt*) malloc(sizeof(PetscInt) * (iend-istart));
    for (unsigned int i = 0; i < (iend-istart); ++i)
    {
        row_indices[i] = i+istart;
    }

    // Retrieving eigenpairs and filling structures
    for (unsigned int i = 0; i < num_eigenpairs; ++i)
    {
        EPSGetEigenpair(eps, i, &eigval, NULL, eigvec, NULL);

        if (eigenvectors)
        {
            VecGetValues(eigvec, iend-istart, row_indices, values);
            MatSetValues(*eigenvectors, iend-istart, row_indices, 1, (int*)&i, values, INSERT_VALUES);
        }
        if (eigenvalues)
        {
            MatSetValues(*eigenvalues, 1, (int*)&i, 1, (int*)&i, &eigval, INSERT_VALUES); // Diagonal
        }

        if (eigenvalues_inv)
        {
            eigval_inv = 1./eigval;
            MatSetValues(*eigenvalues_inv, 1, (int*)&i, 1, (int*)&i, &eigval_inv, INSERT_VALUES); // Diagonal 1/eigval
        }
    }
    VecDestroy(&eigvec);
    free(row_indices);
    free(values);
    EPSDestroy(&eps);

    // Assemble all matrices
    if (eigenvectors)
    {
        MatAssemblyBegin(*eigenvectors, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(*eigenvectors, MAT_FINAL_ASSEMBLY);
    }
    if (eigenvalues)
    {
        MatAssemblyBegin(*eigenvalues, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(*eigenvalues, MAT_FINAL_ASSEMBLY);
    }
    if (eigenvalues_inv)
    {
        MatAssemblyBegin(*eigenvalues_inv, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(*eigenvalues_inv, MAT_FINAL_ASSEMBLY);
    }
}

void EigendecompositionLargest(Mat A, const unsigned int num_eigenpairs, Mat* eigenvectors, Mat* eigenvalues, Mat* eigenvalues_inv)
{
    Eigendecomposition(A, num_eigenpairs, eigenvectors, eigenvalues, eigenvalues_inv, EPS_LARGEST_MAGNITUDE);
}

void EigendecompositionSmallest(Mat A, const unsigned int num_eigenpairs, Mat* eigenvectors, Mat* eigenvalues, Mat* eigenvalues_inv)
{
    Eigendecomposition(A, num_eigenpairs, eigenvectors, eigenvalues, eigenvalues_inv, EPS_SMALLEST_MAGNITUDE);
}
