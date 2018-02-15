#include "gram_schmidt.h"

#include "utils.h"

/*
Projection operator: proj_u(v) = (<u, v> / <u, u>) * u
<u, v> = u^T*v (dot product)
Input: v, u vectors, res should have been created
Output: an allocated and filled res
*/
static void Projection(const Vec v, const Vec u, Vec* res)
{
    PetscScalar val1, val2, factor;
    VecDot(v, u, &val1);
    VecDot(u, u, &val2);
    factor = val1 / val2;

    VecCopy(u, *res);

    VecScale(*res, factor);
}

/*
Orthonormalise a basis of p column vectors
Uses Gram-Schimdt
Input: X is already created and initialised, p the number of cols of X, n number of rows of X, an allocated norms vector of size p
Output: Orthonormal X and norms filled with the norms of X before normalisation
*/
void OrthonormaliseVecs(Vec* X, const unsigned int n, const unsigned int p, PetscScalar* norms)
{
    Vec sum_vec, proj_vec;

    VecCreate(PETSC_COMM_WORLD, &sum_vec);
    VecSetSizes(sum_vec, PETSC_DECIDE, n);
    VecSetFromOptions(sum_vec);
    VecAssemblyBegin(sum_vec);
    VecAssemblyEnd(sum_vec);

    VecDuplicate(sum_vec, &proj_vec);
    VecSetFromOptions(proj_vec);
    VecAssemblyBegin(proj_vec);
    VecAssemblyEnd(proj_vec);

    PetscScalar* current_norm = NULL;
    for (unsigned int k = 0; k < p; ++k)
    {
        VecSet(sum_vec, 0);
        for (unsigned int j = 0; j < k; ++j)
        {
            Projection(X[k], X[j], &proj_vec);
            VecAXPY(sum_vec, 1., proj_vec);
        }
        VecAXPBY(X[k], -1., 1, sum_vec);  // u_k = 1*v_k - sum

        if (norms)
        {
            current_norm = norms+k;
        }
        VecNormalize(X[k], current_norm);
    }

    VecDestroy(&proj_vec);
    VecDestroy(&sum_vec);
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
