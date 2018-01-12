#include "nystroem.h"

#include <stdlib.h>

Mat Nystroem(Mat B, Mat phi_A, Mat Pi_A_Inv, const unsigned int N, const unsigned n, const unsigned p)
{
    Mat phi;
    MatCreate(PETSC_COMM_WORLD, &phi);
    MatSetSizes(phi, PETSC_DECIDE, PETSC_DECIDE, N, p);
    MatSetType(phi, MATMPIDENSE);
    MatSetFromOptions(phi);
    MatSetUp(phi);

    // Fill upper part (each node fills a part of the matrix)
    PetscInt istart, iend;
    PetscInt *col_indices, *row_indices;
    PetscScalar *values;

    col_indices = (PetscInt*) malloc(sizeof(PetscInt) * p);
    for (unsigned int i = 0; i < p; ++i)
    {
        col_indices[i] = i;
    }

    MatGetOwnershipRange(phi_A, &istart, &iend);
    values = (PetscScalar*) malloc(sizeof(PetscScalar) * p * (iend-istart));
    row_indices = (PetscInt*) malloc(sizeof(PetscInt) * (iend-istart));
    for (unsigned int i = 0; i < (iend-istart); ++i)
    {
        row_indices[i] = istart+i;
    }

    MatGetValues(phi_A, iend-istart, row_indices, p, col_indices, values);
    MatSetValues(phi, iend-istart, row_indices, p, col_indices, values, INSERT_VALUES);

    free(values);
    free(row_indices);

    // Fill lower part (each node fills a part of the matrix)
    Mat lower, part_lower;
    MatTransposeMatMult(B, phi_A, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &part_lower);
    MatMatMult(part_lower, Pi_A_Inv, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &lower);
    MatDestroy(&part_lower);
    MatView(lower, PETSC_VIEWER_STDOUT_WORLD);

    MatGetOwnershipRange(lower, &istart, &iend);
    values = (PetscScalar*) malloc(sizeof(PetscScalar) * p * (iend-istart));
    row_indices = (PetscInt*) malloc(sizeof(PetscInt) * (iend-istart));
    for (unsigned int i = 0; i < (iend-istart); ++i)
    {
        row_indices[i] = istart+i;
    }
    MatGetValues(lower, iend-istart, row_indices, p, col_indices, values);
    for (unsigned int i = 0; i < (iend-istart); ++i)
    {
        row_indices[i] = istart+i+n;
    }
    MatSetValues(phi, iend-istart, row_indices, p, col_indices, values, INSERT_VALUES);
    MatDestroy(&lower);

    free(values);
    free(row_indices);
    free(col_indices);

    // Assemble matrix
    MatAssemblyBegin(phi, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(phi, MAT_FINAL_ASSEMBLY);

    return phi;
}
