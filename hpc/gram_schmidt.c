#include "gram_schmidt.h"

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

    // VecScale is not collective, so put factor in Vector and do PointwiseMult
    Vec factor_vec;
    VecDuplicate(*res, &factor_vec);
    VecSetFromOptions(factor_vec);
    VecSet(factor_vec, factor);
    VecAssemblyBegin(factor_vec);
    VecAssemblyEnd(factor_vec);
    VecPointwiseMult(*res, *res, factor_vec);
    VecDestroy(&factor_vec);
}

/*
Orthonormalise a basis of p column vectors
Uses Gram-Schimdt
Input: X is already created and initialised, p the number of cols of X, n number of rows of X
Output: Orthonormal X
*/
void OrthonormaliseVecs(Vec* X, const unsigned int n, const unsigned int p)  
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

    for (unsigned int k = 0; k < p; ++k)
    {
        VecSet(sum_vec, 0);
        for (unsigned int j = 0; j < k; ++j)
        {
            Projection(X[k], X[j], &proj_vec);
            VecAXPY(sum_vec, 1., proj_vec);
        }
        VecAXPBY(X[k], -1., 1, sum_vec);  // u_k = 1*v_k - sum
        VecNormalize(X[k], NULL);
    }

    VecDestroy(&proj_vec);
    VecDestroy(&sum_vec);
}