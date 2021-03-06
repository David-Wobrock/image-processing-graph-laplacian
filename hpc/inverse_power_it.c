#include "inverse_power_it.h"

#include <petscvec.h>
#include <petscksp.h>

#include "utils.h"
#include "gram_schmidt.h"

/* Returns an array of m Vec instances, each Vec of size p
They are filled with random values, assembled and initialised
*/
static Vec* BuildRandomVectors(const unsigned int p, const unsigned int m)
{
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    // Build X_0 as array of column vectors
    Vec* X_k = (Vec*) malloc(sizeof(Vec) * m);
    for (unsigned int i = 0; i < m; ++i)
    {
        VecCreate(PETSC_COMM_WORLD, &(X_k[i]));
        VecSetSizes(X_k[i], PETSC_DECIDE, p);
        VecSetFromOptions(X_k[i]);
    }

    // Fill with random, seed=rank
    PetscRandom rand_ctx;
    PetscRandomCreate(PETSC_COMM_SELF, &rand_ctx);
    PetscRandomSetSeed(rand_ctx, rank);
    PetscRandomSeed(rand_ctx);
    for (unsigned int i = 0; i < m; ++i)
    {
        VecSetRandom(X_k[i], rand_ctx);
    }
    PetscRandomDestroy(&rand_ctx);

    for (unsigned int i = 0; i < m; ++i)
    {
        VecAssemblyBegin(X_k[i]);
    }
    for (unsigned int i = 0; i < m; ++i)
    {
        VecAssemblyEnd(X_k[i]);
    }

    return X_k;
}

static PetscScalar ComputeResidualsNorm(Mat A, Vec* X_k_vec, const unsigned int m)
{
    Mat Xk;
    Vecs2Mat(X_k_vec, &Xk, m);

    Mat XkT;
    MatTranspose(Xk, MAT_INITIAL_MATRIX, &XkT);

    Mat XkXkT;
    MatMatMult(Xk, XkT, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &XkXkT);
    // MatMatTransposeMult(Xk, Xk, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &XkXkT); // not supported for MPIDense (state for petsc-3.8.3)

    // I - XkXkT => -(XkXkT - I)
    MatShift(XkXkT, -1);  // Y = Y + a*I
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
void InversePowerIteration(const Mat A, const unsigned int m, Mat* eigenvectors, Mat* eigenvalues, PetscBool optiGramSchmidt, PetscScalar epsilon)
{
    PetscInt p;
    MatGetSize(A, &p, NULL);

    PetscScalar* norms = (PetscScalar*) malloc(sizeof(PetscScalar) * m);

    // Build X_0, initial random vector
    Vec* X_k = BuildRandomVectors(p, m);
    OrthonormaliseVecs(X_k, p, m, norms);

    Vec* X_k_before_orth = (Vec*) malloc(sizeof(Vec) * m);
    for (unsigned int i = 0; i < m; ++i)
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
    PetscScalar r_norm;

    KSP ksp;
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, A, A);
    KSPSetType(ksp, KSPGMRES);

    PC pc;
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCASM);

    PCASMSetLocalSubdomains(pc, 2, NULL, NULL);  // Number of subdomains per proc
    PCASMSetType(pc, PC_ASM_RESTRICT);
    PCASMSetOverlap(pc, 0); // Overlap

    KSPSetFromOptions(ksp);
    KSPSetUp(ksp);

    PCType type;
    PCGetType(pc, &type);
    PetscPrintf(PETSC_COMM_WORLD, "TYPE %s\n", type);

    // Set subsolvers
    if (strcmp(type, PCASM) == 0)
    {
        KSP *subksp;
        PC subpc;
        PetscInt num_local;
        PCASMGetSubKSP(pc, &num_local, NULL, &subksp);
        for (unsigned int i = 0; i < num_local; ++i)
        {
            KSPSetType(subksp[i], KSPGMRES);

            KSPGetPC(subksp[i], &subpc);
            PCSetType(subpc, PCNONE);

            KSPSetFromOptions(subksp[i]);
        }
    }

    r_norm = ComputeResidualsNorm(A, X_k, m);
    unsigned int num_outer_it = 0;
    while (r_norm > epsilon)
    {
        ++num_outer_it;
        double solve_start = MPI_Wtime();
        for (unsigned int i = 0; i < m; ++i)
        {
            KSPSolve(ksp, X_k[i], X_k[i]);
        }
        PetscPrintf(PETSC_COMM_WORLD, "* Solving %d systems took %fs\n", m, MPI_Wtime() - solve_start);
        // Save X_k before orthonormalisation
        CopyVecs(X_k, X_k_before_orth, m);

        double ortho_start = MPI_Wtime();
        if (num_outer_it % optiGramSchmidt == 0)
        {
            OrthonormaliseVecs(X_k, p, m, norms);
        }
        PetscPrintf(PETSC_COMM_WORLD, "* Orthonormalising took %fs\n", MPI_Wtime() - ortho_start);
        double res_start = MPI_Wtime();
        r_norm = ComputeResidualsNorm(A, X_k, m);
        PetscPrintf(PETSC_COMM_WORLD, "* Computing residual %fs (outer iteration %d - residual %f)\n***\n", MPI_Wtime() - res_start, num_outer_it, r_norm);
    }
    if (optiGramSchmidt != 1 && (num_outer_it % optiGramSchmidt) != 0)
    {
        OrthonormaliseVecs(X_k, p, m, norms);
    }
    PetscPrintf(PETSC_COMM_WORLD, "Inverse subspace iteration took %d outer iterations\n", num_outer_it);
    KSPDestroy(&ksp);

    // Compute eigenvalues
    if (eigenvalues)
    {
        MatCreate(PETSC_COMM_WORLD, eigenvalues);
        MatSetSizes(*eigenvalues, PETSC_DECIDE, PETSC_DECIDE, m, m);
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
        MatSetSizes(*eigenvectors, PETSC_DECIDE, PETSC_DECIDE, p, m);
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
        NormaliseVecs(X_k_before_orth, m, NULL);
        for (PetscInt i = 0; i < m; ++i)
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
    for (unsigned int i = 0; i < m; ++i)
    {
        VecDestroy(&(X_k[i]));
        VecDestroy(&(X_k_before_orth[i]));
    }
    free(X_k);
    free(X_k_before_orth);
}
