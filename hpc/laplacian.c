#include "laplacian.h"

#include <petscvec.h>
#include "utils.h"

/*
Lapl = alpha * (D-K)
 D_A = rowSum([K_A K_B])
 alpha = 1./ mean(D)
So,
 Lapl_A = alpha * (D_A - K_A)
 Lapl_B = -alpha*K_B
*/
void ComputeLaplacianMatrix(Mat* L_A, Mat* L_B, Mat K_A, Mat K_B)
{
    // D_A
    Vec D_vec, D_vecB;
    D_vec = MatRowSum(K_A);
    D_vecB = MatRowSum(K_B);
    VecAXPY(D_vec, 1.0, D_vecB);
    VecDestroy(&D_vecB);

    Mat D_A;
    MatDuplicate(K_A, MAT_DO_NOT_COPY_VALUES, &D_A);
    MatZeroEntries(D_A);
    MatDiagonalSet(D_A, D_vec, INSERT_VALUES);

    // alpha
    PetscScalar alpha = 1.0 / VecMean(D_vec);
    VecDestroy(&D_vec);

    // Compute L_A
    MatDuplicate(K_A, MAT_COPY_VALUES, L_A);
    MatAYPX(*L_A, -1.0, D_A, SAME_NONZERO_PATTERN);
    MatScale(*L_A, alpha);

    // Compute L_B
    MatDuplicate(K_B, MAT_COPY_VALUES, L_B);
    MatScale(*L_B, -alpha);

    MatDestroy(&D_A);
}

void ComputeEntireLaplacianMatrix(Mat* Lapl, Mat K)
{
    // D
    Vec D_vec;
    D_vec = MatRowSum(K);

    Mat D;
    MatDuplicate(K, MAT_DO_NOT_COPY_VALUES, &D);
    MatZeroEntries(D);
    MatDiagonalSet(D, D_vec, INSERT_VALUES);

    // alpha
    PetscScalar alpha = 1.0 / VecMean(D_vec);
    VecDestroy(&D_vec);

    // Compute Lapl
    MatDuplicate(K, MAT_COPY_VALUES, Lapl);
    MatAYPX(*Lapl, -1.0, D, SAME_NONZERO_PATTERN); // Y = a*Y + X
    MatScale(*Lapl, alpha);

    MatDestroy(&D);
}
