#include "sinkhorn.h"

#include "utils.h"

static void UpdateVector(Mat v, Mat x, Mat phi, Mat Pi, Mat phi_T)
{
    Mat right_part, left_part, denominator, res;

    // Calc v = 1. / phi*Pi*phi_T*x;
    MatMatMult(phi, Pi, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &left_part);
    MatMatMult(phi_T, x, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &right_part);
    MatMatMult(left_part, right_part, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &denominator);

    // Create another matrix for inverse 1/x
    MatDuplicate(v, MAT_DO_NOT_COPY_VALUES, &res);
    PetscInt istart, iend;
    MatGetOwnershipRange(denominator, &istart, &iend);
    const PetscScalar* values;
    PetscScalar inv_values;
    for (PetscInt i = istart; i < iend; ++i)
    {
        MatGetRow(denominator, i, NULL, NULL, &values);
        inv_values = 1./values[0];
        MatSetValues(res, 1, &i, 1, &ZERO, &inv_values, INSERT_VALUES);
        MatRestoreRow(denominator, i, NULL, NULL, &values);
    }
    MatAssemblyBegin(res, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(res, MAT_FINAL_ASSEMBLY);

    // Copy into v
    MatCopy(res, v, SAME_NONZERO_PATTERN);

    MatDestroy(&res);
    MatDestroy(&denominator);
    MatDestroy(&right_part);
    MatDestroy(&left_part);
}

/*
r of size Nx1
c of size Nx1
W_A and W_B should not be initialised
*/
static void ComputeWAWB(Mat phi, Mat Pi, Mat phi_T, Mat r, Mat c, Mat* W_A, Mat* W_B, const unsigned int N, const unsigned int p)
{
    // WAWB = R*K*C = R*phi*Pi*phi_T*C, but only p first rows are necessary

    // R*phi (first p rows)
    Mat r_p = GetFirstRows(r, p);
    Mat diag_r_p = OneColMat2Diag(r_p);
    MatDestroy(&r_p);
    Mat phi_p = GetFirstRows(phi, p);
    Mat r_phi;
    MatMatMult(diag_r_p, phi_p, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &r_phi);
    MatDestroy(&phi_p);
    MatDestroy(&diag_r_p);

    // r_phi * Pi
    Mat r_phi_Pi;
    MatMatMult(r_phi, Pi, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &r_phi_Pi);
    MatDestroy(&r_phi);
    
    // phi_T * C
    Mat diag_c = OneColMat2Diag(c);
    Mat phi_T_c;
    MatMatMult(phi_T, diag_c, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &phi_T_c);
    MatDestroy(&diag_c);

    // Compute W_AW_B
    Mat WAWB;
    MatMatMult(r_phi_Pi, phi_T_c, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &WAWB);
    MatDestroy(&phi_T_c);
    MatDestroy(&r_phi_Pi);

    // Separate WAWB into W_A & W_B
    *W_A = GetFirstCols(WAWB, p);
    *W_B = GetLastCols(WAWB, N-p);
}

void Sinkhorn(Mat phi, Mat Pi, Mat* W_A, Mat* W_B)
{
    PetscInt N, p;
    MatGetSize(phi, &N, &p);

    // Initialise r and c
    Mat r, c;
    MatCreate(PETSC_COMM_WORLD, &r);
    MatSetSizes(r, PETSC_DECIDE, PETSC_DECIDE, N, 1);
    MatSetType(r, MATMPIDENSE);
    MatSetFromOptions(r);
    MatSetUp(r);

    // Fill r with ones (each process fills local part)
    PetscInt istart, iend, num_rows, *row_indices;
    PetscScalar* ones;
    MatGetOwnershipRange(r, &istart, &iend);
    num_rows = iend - istart;
    row_indices = (PetscInt*) malloc(sizeof(PetscInt) * num_rows);
    ones = (PetscScalar*) malloc(sizeof(PetscScalar) * num_rows);
    for (unsigned int i = 0; i < num_rows; ++i)
    {
        row_indices[i] = istart+i;
        ones[i] = 1;
    }
    MatSetValues(r, num_rows, row_indices, 1, &ZERO, ones, INSERT_VALUES);

    MatDuplicate(r, MAT_DO_NOT_COPY_VALUES, &c);

    MatAssemblyBegin(r, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(c, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(r, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(c, MAT_FINAL_ASSEMBLY);

    free(ones);
    free(row_indices);

    // Compute phi_T
    Mat phi_T;
    MatTranspose(phi, MAT_INITIAL_MATRIX, &phi_T);

    // Compute r and c vectors iteratively
    for (unsigned int i = 0; i < 100; ++i)
    {
        UpdateVector(c, r, phi, Pi, phi_T);
        UpdateVector(r, c, phi, Pi, phi_T);
    }

    // Compute W_A and W_B
    ComputeWAWB(phi, Pi, phi_T, r, c, W_A, W_B, N, p);

    MatDestroy(&phi_T);
    MatDestroy(&c);
    MatDestroy(&r);
}
