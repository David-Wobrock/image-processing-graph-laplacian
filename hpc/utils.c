#include "utils.h"

unsigned int num2x(const unsigned int num, const unsigned int num_col)
{
    return (int) (num/num_col);
}

unsigned int num2y(const unsigned int num, const unsigned int num_col)
{
    return num % num_col;
}

/* Pass uninitialised matrix
Done in COMM_WORLD
*/
void Vecs2Mat(Vec* vecs, Mat* m, const unsigned int ncols)
{
    PetscInt nrows;
    VecGetSize(vecs[0], &nrows);

    //MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, nrows, ncols, NULL, m);
    MatCreate(PETSC_COMM_WORLD, m);
    MatSetSizes(*m, PETSC_DECIDE, PETSC_DECIDE, nrows, ncols);
    MatSetType(*m, MATMPIDENSE);
    MatSetFromOptions(*m);
    MatSetUp(*m);

    PetscInt start, end;
    PetscInt* indices;
    PetscScalar* values;
    for (unsigned int i = 0; i < ncols; ++i)
    {
        VecGetOwnershipRange(vecs[i], &start, &end);
        indices = (PetscInt*) malloc(sizeof(PetscInt) * (end-start));
        for (unsigned int j = 0; j < end-start; ++j)
        {
            indices[j] = start+j;
        }
        values = (PetscScalar*) malloc(sizeof(PetscScalar) * (end-start));

        VecGetValues(vecs[i], end-start, indices, values);
        MatSetValues(*m, end-start, indices, 1, (int*)&i, values, ADD_VALUES);

        free(values);
        free(indices);
    }

    MatAssemblyBegin(*m, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*m, MAT_FINAL_ASSEMBLY);
}
