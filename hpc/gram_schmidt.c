#include "gram_schmidt.h"

#include "utils.h"

/*
Projection operator: proj_u(v) = (<u, v> / <u, u>) * u
<u, v> = u^T*v (dot product)
Input: v, u vectors, res should have been created
Output: an allocated and filled res
*/
/*static void Projection(const Vec v, const Vec u, Vec* res)
{
    PetscScalar val1, val2, factor;
    VecDot(v, u, &val1);
    VecDot(u, u, &val2);
    factor = val1 / val2;

    VecCopy(u, *res);

    VecScale(*res, factor);
}*/

static double LocalVecDot(Vec x, Vec y)
{
    PetscInt istart, iend;
    VecGetOwnershipRange(x, &istart, &iend);

    double dot = 0;
    double val_x, val_y;
    for (PetscInt i = istart; i < iend; ++i)
    {
        VecGetValues(x, 1, &i, &val_x);
        VecGetValues(y, 1, &i, &val_y);
        dot += (val_x * val_y);
    }

    return dot;
}

/*
Orthonormalise a basis of p column vectors
Uses Gram-Schimdt
Input: X is already created and initialised, p the number of cols of X, n number of rows of X, an allocated norms vector of size p
Output: Orthonormal X and norms filled with the norms of X before normalisation
*/
void OrthonormaliseVecs(Vec* X, const unsigned int n, const unsigned int p, PetscScalar* norms)
{
    Vec X_j_copy;
    VecCreate(PETSC_COMM_WORLD, &X_j_copy);
    VecSetSizes(X_j_copy, PETSC_DECIDE, n);
    VecSetFromOptions(X_j_copy);
    VecAssemblyBegin(X_j_copy);
    VecAssemblyEnd(X_j_copy);

    PetscScalar* current_norm = NULL;
    unsigned int k, j;
    double factor;
    double *dot_product_sums, *comm_sums;
    dot_product_sums = (double*) malloc(sizeof(double) * p * 2);
    comm_sums = (double*) malloc(sizeof(double) * p * 2);
    for (k = 0; k < p; ++k)
    {
        // Local sum or projections
        for (j = 0; j < k; ++j)
        {
            dot_product_sums[j*2] = LocalVecDot(X[j], X[k]);
            dot_product_sums[(j*2)+1] = LocalVecDot(X[j], X[j]);
        }

        // AllReduce sums
        MPI_Allreduce(
            dot_product_sums,
            comm_sums,
            k*2,
            MPI_DOUBLE,
            MPI_SUM,
            PETSC_COMM_WORLD);

        // Apply orthogonalisation locally
        for (j = 0; j < k; ++j)
        {
            // X_k = X_k - (a/b) * X_j
            VecCopy(X[j], X_j_copy);
            factor = ((double) comm_sums[j*2]) / (comm_sums[(j*2)+1]);
            VecScale(X_j_copy, factor); // Not collective

            VecAXPY(X[k], -1.0, X_j_copy);
        }

        // Normalise
        if (norms)
        {
            current_norm = norms+k;
        }
        VecNormalize(X[k], current_norm);
    }

    free(comm_sums);
    free(dot_product_sums);
    VecDestroy(&X_j_copy);
}

void NormaliseVecs(Vec* X, const unsigned int p, PetscScalar* norms)
{
    PetscScalar* current_norm = NULL;
    for (unsigned int i = 0; i < p; ++i)
    {
        if (norms)
        {
            current_norm = norms+i;
        }
        VecNormalize(X[i], current_norm);
    }
}
