#include "laplacian.h"

#include <petscvec.h>
#include "utils.h"

void ComputeLaplacianMatrix(Mat* L_A, Mat K_A)
{
    PetscInt p;
    MatGetSize(K_A, &p, NULL);

    // D^{-1/2}
    Vec D_vec;
    D_vec = MatRowSum(K_A);
    VecSqrtAbs(D_vec);
    VecPow(D_vec, -1);
    Mat D_diag;
    MatDuplicate(K_A, MAT_DO_NOT_COPY_VALUES, &D_diag);
    MatZeroEntries(D_diag);
    MatDiagonalSet(D_diag, D_vec, INSERT_VALUES);
    VecDestroy(&D_vec);

    // Compute L_A
    Mat Id = MatCreateIdentity(p, MATMPIDENSE);
    Mat right_part, tmp;
    MatMatMult(D_diag, K_A, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &tmp);
    MatMatMult(tmp, D_diag, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &right_part);
    MatDestroy(&tmp);
    MatDestroy(&D_diag);

    MatScale(right_part, -1.0);

    Mat matrices[2];
    matrices[0] = Id;
    matrices[1] = right_part;
    MatCreateComposite(PETSC_COMM_WORLD, 2, matrices, L_A);
    MatCompositeMerge(*L_A);

    MatDestroy(&right_part);
    MatDestroy(&Id);
}
