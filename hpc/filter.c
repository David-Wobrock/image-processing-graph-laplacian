#include "filter.h"

#include "utils.h"

static Mat ComputeK(Mat phi, Mat Pi)
{
    PetscInt p;
    MatGetSize(phi, NULL, &p);

    Mat phi_firstrows = GetFirstRows(phi, p);
    Mat left_part;
    MatMatMult(phi_firstrows, Pi, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &left_part);
    MatDestroy(&phi_firstrows);

    Mat phi_T;
    MatTranspose(phi, MAT_INITIAL_MATRIX, &phi_T);

    Mat K;
    MatMatMult(left_part, phi_T, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &K);
    MatDestroy(&left_part);
    MatDestroy(&phi_T);

    return K;
}

/*
Pass uninitialised W_A and W_B
Re-normalised Laplacian: L=alpha*(D-K)
Filter: W=I-L <=> W=I-alpha*(D-K) <=> W=I+alpha*(K-D)
*/
void ComputeWAWB_RenormalisedLaplacian(Mat phi, Mat Pi, Mat* W_A, Mat* W_B)
{
    Mat K = ComputeK(phi, Pi);
    PetscInt p, N;
    MatGetSize(K, &p, &N);

    Vec D;
    VecCreate(PETSC_COMM_WORLD, &D);
    VecSetSizes(D, PETSC_DECIDE, p);
    VecSetFromOptions(D);
    MatGetRowSum(K, D);

    // K-D
    Mat KminusD;
    MatCreate(PETSC_COMM_WORLD, &KminusD);
    MatSetSizes(KminusD, PETSC_DECIDE, PETSC_DECIDE, p, N);
    MatSetType(KminusD, MATMPIDENSE);
    MatSetFromOptions(KminusD);
    MatSetUp(KminusD);

    PetscInt ncols;
    const PetscInt *col_indices;
    const PetscScalar *K_values;
    PetscScalar *values, d;
    values = (PetscScalar*) malloc(sizeof(PetscScalar) * N);
    PetscInt istart, iend;
    MatGetOwnershipRange(K, &istart, &iend);
    for (PetscInt i = istart; i < iend; ++i)
    {
        MatGetRow(K, i, &ncols, &col_indices, &K_values);
        VecGetValues(D, 1, &i, &d);

        memcpy(values, K_values, sizeof(PetscScalar) * N);
        values[i] -= d;
        MatSetValues(KminusD, 1, &i, ncols, col_indices, values, INSERT_VALUES);

        MatRestoreRow(K, i, &ncols, &col_indices, &K_values);
    }
    free(values);
    MatAssemblyBegin(KminusD, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(KminusD, MAT_FINAL_ASSEMBLY);

    // alpha*(K-D)
    PetscScalar alpha = 1. / VecMean(D);
    MatScale(KminusD, alpha);
    VecDestroy(&D);
    MatDestroy(&K);

    // Split W_A and W_B
    Mat W_A_without_ident = GetFirstCols(KminusD, p);
    *W_B = GetLastCols(KminusD, N-p);
    MatDestroy(&KminusD);

    // I + alpha*(K - D) (only affects W_A)
    Mat ident = MatCreateIdentity(p, MATMPIDENSE);
    Mat matrices[2];
    matrices[0] = ident;
    matrices[1] = W_A_without_ident;
    MatCreateComposite(PETSC_COMM_WORLD, 2, matrices, W_A);
    MatCompositeMerge(*W_A);

    MatDestroy(&ident);
    MatDestroy(&W_A_without_ident);
}
