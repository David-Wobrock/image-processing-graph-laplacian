#include <petscmat.h>
#include <petscvec.h>
#include <petscksp.h>

#include "utils.h"
#include "gram_schmidt.h"

/* Returns an array of p Vec instances.
They are filled with random values, assembled and initialised
*/
static Vec* BuildRandomVectors(const unsigned int n, const unsigned int p)
{
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    // Build X_0 as array of column vectors
    Vec* X_k = (Vec*) malloc(sizeof(Vec) * p);
    for (unsigned int i = 0; i < p; ++i)
    {
        VecCreate(PETSC_COMM_WORLD, &(X_k[i]));
        VecSetSizes(X_k[i], PETSC_DECIDE, n);
        VecSetFromOptions(X_k[i]);
    }

    // Fill with random, seed=rank
    PetscRandom rand_ctx;
    PetscRandomCreate(PETSC_COMM_SELF, &rand_ctx);
    PetscRandomSetSeed(rand_ctx, rank);
    PetscRandomSeed(rand_ctx);
    for (unsigned int i = 0; i < p; ++i)
    {
        VecSetRandom(X_k[i], rand_ctx);
    }
    PetscRandomDestroy(&rand_ctx);

    for (unsigned int i = 0; i < p; ++i)
    {
        VecAssemblyBegin(X_k[i]);
    }
    for (unsigned int i = 0; i < p; ++i)
    {
        VecAssemblyEnd(X_k[i]);
    }

    return X_k;
}

static PetscScalar ComputeResidualsNorm(Mat A, Vec* X_k_vec, const unsigned int p)
{
    Mat Xk;
    Vecs2Mat(X_k_vec, &Xk, p);

    Mat XkT;
    MatTranspose(Xk, MAT_INITIAL_MATRIX, &XkT);

    Mat XkXkT;
    MatMatMult(Xk, XkT, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &XkXkT);
    // MatMatTransposeMult(Xk, Xk, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &XkXkT); // not supported for MPIDense (state for petsc-3.8.3)

    // I - XkXkT => -(XkXkT - I)
    MatShift(XkXkT, -1);
    MatScale(XkXkT, -1);

    // Rk = (I - XkXkT) A Xk
    Mat Rk, tmp;
    MatMatMult(XkXkT, A, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &tmp);
    MatMatMult(tmp, Xk, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Rk);
    //MatMatMatMult(XkXkT, A, Xk, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Rk); not supported for MPIDense*3 (state for petsc-3.8.3)

    PetscScalar norm;
    MatNorm(Rk, NORM_FROBENIUS, &norm);

    MatDestroy(&Rk);
    MatDestroy(&tmp);
    MatDestroy(&XkXkT);
    MatDestroy(&XkT);
    MatDestroy(&Xk);
    return norm;
}

/*
A is symmetric (and square)
p the number of eigenvectors
*/
void InversePowerIteration(const Mat A, const unsigned int p, Mat* eigenvectors, Mat* eigenvalues)
{
    PetscInt n;
    MatGetSize(A, &n, NULL);

    PetscScalar* norms = (PetscScalar*) malloc(sizeof(PetscScalar) * p);

    // Build X_0, initial random vector
    Vec* X_k = BuildRandomVectors(n, p);
    OrthonormaliseVecs(X_k, n, p, norms);

    Vec* X_k_before_orth = (Vec*) malloc(sizeof(Vec) * p);
    for (unsigned int i = 0; i < p; ++i)
    {
        VecDuplicate(X_k[i], X_k_before_orth+i);
    }

    // Test if X_k, k=0 is orthonormal
    /*Mat X_k_mat, res;
    Vecs2Mat(X_k, &X_k_mat, p);
    MatTransposeMatMult(X_k_mat, X_k_mat, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &res);
    MatView(res, PETSC_VIEWER_STDOUT_WORLD); // Should be identity
    MatDestroy(&res);
    MatDestroy(&X_k_mat);

    PetscReal norm;
    for (unsigned int i = 0; i < p; ++i)
    {
        VecNorm(X_k[i], NORM_2, &norm);
        PetscPrintf(PETSC_COMM_WORLD, "Vec %d: %f\n", i, norm); // Should all be 1
    }*/

    // Inverse subspace/power iteration
    PetscScalar epsilon = 0.1, r_norm;

    KSP ksp;
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, A, A);
    KSPSetType(ksp, KSPGMRES);

    PC pc;
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCASM);

    PCASMSetLocalSubdomains(pc, 2, NULL, NULL);  // Number of subdomains per proc
    PCASMSetType(pc, PC_ASM_BASIC);
    PCASMSetOverlap(pc, 0); // Overlap

    KSPSetFromOptions(ksp);
    KSPSetUp(ksp);

    // Set subsolvers
    KSP *subksp;
    PetscInt num_local;
    PCASMGetSubKSP(pc, &num_local, NULL, &subksp);
    for (unsigned int i = 0; i < num_local; ++i)
    {
        KSPSetType(subksp[i], KSPGMRES);
    }

    r_norm = ComputeResidualsNorm(A, X_k, p);
    while (r_norm > epsilon)
    {
        for (unsigned int i = 0; i < p; ++i)
        {
            KSPSolve(ksp, X_k[i], X_k[i]);
        }
        // Save X_k before orthonormalisation
        CopyVecs(X_k, X_k_before_orth, p);

        OrthonormaliseVecs(X_k, n, p, norms);
        r_norm = ComputeResidualsNorm(A, X_k, p);
        PetscPrintf(PETSC_COMM_WORLD, "New residual: %f\n", r_norm);
    }
    KSPDestroy(&ksp);

    // Compute eigenvalues
    if (eigenvalues)
    {
        MatCreate(PETSC_COMM_WORLD, eigenvalues);
        MatSetSizes(*eigenvalues, PETSC_DECIDE, PETSC_DECIDE, p, p);
        MatSetType(*eigenvalues, MATMPIAIJ);
        MatSetFromOptions(*eigenvalues);
        MatSetUp(*eigenvalues);

        PetscScalar value;
        PetscInt istart, iend;
        MatGetOwnershipRange(*eigenvalues, &istart, &iend);
        for (PetscInt i = istart; i < iend; ++i)
        {
            value = 1.0 / norms[i];
            MatSetValues(*eigenvalues, 1, &i, 1, &i, &value, INSERT_VALUES);
        }

        MatAssemblyBegin(*eigenvalues, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(*eigenvalues, MAT_FINAL_ASSEMBLY);
    }

    // Compute eigenvectors
    if (eigenvectors)
    {
        MatCreate(PETSC_COMM_WORLD, eigenvectors);
        MatSetSizes(*eigenvectors, PETSC_DECIDE, PETSC_DECIDE, n, p);
        MatSetType(*eigenvectors, MATMPIDENSE);
        MatSetFromOptions(*eigenvectors);
        MatSetUp(*eigenvectors);

        PetscInt istart, iend;
        VecGetOwnershipRange(X_k[0], &istart, &iend);
        PetscInt* indices = (PetscInt*) malloc(sizeof(PetscInt) * (iend-istart));
        for (unsigned int i = 0; i < (iend-istart); ++i)
        {
            indices[i] = i+istart;
        }
        PetscScalar* values = (PetscScalar*) malloc(sizeof(PetscScalar) * (iend-istart));
        // Each process fills a part of the current vector (istart & iend should be the same for all vectors)
        NormaliseVecs(X_k_before_orth, p, NULL);
        for (PetscInt i = 0; i < p; ++i)
        {
            VecGetValues(X_k_before_orth[i], iend-istart, indices, values);
            MatSetValues(*eigenvectors, iend-istart, indices, 1, &i, values, INSERT_VALUES);
        }
        free(indices);
        free(values);

        MatAssemblyBegin(*eigenvectors, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(*eigenvectors, MAT_FINAL_ASSEMBLY);
    }

    // Free
    free(norms);
    for (unsigned int i = 0; i < p; ++i)
    {
        VecDestroy(&(X_k[i]));
        VecDestroy(&(X_k_before_orth[i]));
    }
    free(X_k);
    free(X_k_before_orth);
}
