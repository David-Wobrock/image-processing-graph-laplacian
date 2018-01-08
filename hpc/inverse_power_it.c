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

PetscReal* InversePowerIteration(const Mat A, const unsigned int n, const unsigned int p)
{
    Vec* X_k = BuildRandomVectors(n, p);
    PetscPrintf(PETSC_COMM_WORLD, "Initial norm %f\n", ComputeResidualsNorm(A, X_k, p));

    OrthonormaliseVecs(X_k, n, p);
    PetscPrintf(PETSC_COMM_WORLD, "After orthonormalisation %f\n", ComputeResidualsNorm(A, X_k, p));

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
    PetscScalar epsilon = 0.01, r_norm;

    KSPType ksptype;
    KSP ksp;
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, A, A);
    //KSPSetType(ksp, KSPGMRES);
    //KSPSetType(ksp, KSPPREONLY);

    //PC pc;
    //KSPGetPC(ksp, &pc);
    //PCSetType(pc, PCASM);
    //PCSetType(pc, PCLU);
    //PCFactorSetMatSolverPackage(pc, MATSOLVERELEMENTAL);

    //PCASMSetTotalSubdomains(pc, size*2, NULL, NULL);
    //PCASMSetType(pc, PC_ASM_RESTRICT);
    //PCASMSetOverlap(pc, 0);

    KSPSetFromOptions(ksp);
    KSPSetUp(ksp);

    // Set subsolvers
    //KSP *subksp;
    //PC subpc;
    //PetscInt num_local;
    //PCASMGetSubKSP(pc, &num_local, NULL, &subksp);
    //for (unsigned int i = 0; i < num_local; ++i)
    //{
    //    KSPSetType(subksp[i], KSPGMRES);
    //    // TODO evaluate number of iterations per domain
    // //   KSPSetType(subksp[i], KSPPREONLY);
    // //   KSPGetPC(subksp[i], &subpc);
    // //   PCSetType(subpc, PCLU);
    //}

    PetscInt num_it;
    PetscReal* norms_before_orth = (PetscReal*) malloc(sizeof(PetscReal) * p);

    double start_inv_it = MPI_Wtime();
    r_norm = ComputeResidualsNorm(A, X_k, p);
    while (r_norm > epsilon)
    {
        PetscPrintf(PETSC_COMM_WORLD, "Residuals norm %f\n", r_norm);
        for (unsigned int i = 0; i < p; ++i)
        {
            KSPSolve(ksp, X_k[i], X_k[i]);
            KSPGetIterationNumber(ksp, &num_it);
            KSPGetType(ksp, &ksptype);
            PetscPrintf(PETSC_COMM_WORLD, "col=%d, method=%s, iterations=%d\n", i, ksptype, num_it);
            VecNorm(X_k[i], NORM_2, norms_before_orth+i);
        }
        OrthonormaliseVecs(X_k, n, p);
        r_norm = ComputeResidualsNorm(A, X_k, p);
    }

    PetscPrintf(PETSC_COMM_WORLD, "Inverse iteration took %f\n", MPI_Wtime() - start_inv_it);
    KSPDestroy(&ksp);
    for (unsigned int i = 0; i < p; ++i)
    {
        VecDestroy(&(X_k[i]));
    }
    free(X_k);

    // Compute eigenvalues
    PetscReal* eigenvalues = norms_before_orth; // Just naming, no reallocation
    for (unsigned int i = 0; i < p; ++i)
    {
        eigenvalues[i] = 1./norms_before_orth[i];
    }
    PetscSortReal(p, eigenvalues);

    return eigenvalues;
}
