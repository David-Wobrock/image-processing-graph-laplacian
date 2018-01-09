#include "utils.h"

unsigned int num2x(const unsigned int num, const unsigned int num_col)
{
    return (int) (num/num_col);
}

unsigned int num2y(const unsigned int num, const unsigned int num_col)
{
    return num % num_col;
}

unsigned int xy2num(const unsigned int x, const unsigned int y, const unsigned int num_col)
{
    return x*num_col + y;
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

Vec* Mat2Vecs(Mat m)
{
    unsigned int i;

    PetscInt n, p;
    MatGetSize(m, &n, &p);

    // Create
    Vec* x = (Vec*) malloc(sizeof(Vec) * p);
    for (i = 0; i < p; ++i)
    {
        VecCreate(PETSC_COMM_WORLD, x+i);
        VecSetSizes(x[i], PETSC_DECIDE, n);
        VecSetFromOptions(x[i]);
    }

    // Fill
    for (i = 0; i < p; ++i)
    {
        MatGetColumnVector(m, x[i], i);
    }

    // Assemble
    for (i = 0; i < p; ++i)
    {
        VecAssemblyBegin(x[i]);
    }
    for (i = 0; i < p; ++i)
    {
        VecAssemblyEnd(x[i]);
    }

    return x;
}

/*
The first rows of the input matrix will be put at the global position specified by
the sample_indices.
*/
Mat Permutation(Mat m, const unsigned int* const sample_indices, const unsigned int num_sample_indices)
{
    Mat reordered;
    MatDuplicate(m, MAT_DO_NOT_COPY_VALUES, &reordered);

    // Fill in correct order
    PetscInt nb_cols, istart, iend, new_pos;
    const PetscInt* col_indices;
    const PetscScalar* values;
    MatGetOwnershipRange(m, &istart, &iend);
    for (PetscInt i = istart; i < iend; ++i)
    {
        MatGetRow(m, i, &nb_cols, &col_indices, &values);

        // Find where this row goes in the new matrix
        if (i < num_sample_indices)
        {
            new_pos = sample_indices[i];
        }
        else
        {
            // Find how many sample indices are before
            unsigned int num_previous_sample_indices = 0;
            while (num_previous_sample_indices < num_sample_indices && sample_indices[num_previous_sample_indices] <= (i - num_sample_indices))
            {
                ++num_previous_sample_indices;
            }
            new_pos = i - num_sample_indices + num_previous_sample_indices;
        }

        MatSetValues(reordered, 1, &new_pos, nb_cols, col_indices, values, INSERT_VALUES);
        MatRestoreRow(m, i, &nb_cols, &col_indices, &values);
    }

    // Assemble
    MatAssemblyBegin(reordered, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(reordered, MAT_FINAL_ASSEMBLY);

    return reordered;
}
