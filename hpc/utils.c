#include "utils.h"

#include <mpi.h>
#include <stdlib.h>

#define min(a, b) (a <= b ? a : b)

const PetscInt ZERO = 0;

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
x is of size Nx1
*/
Mat OneColMat2Diag(Mat x)
{
    PetscInt nb_rows;
    MatGetSize(x, &nb_rows, NULL);

    Mat y;
    MatCreate(PETSC_COMM_WORLD, &y);
    MatSetSizes(y, PETSC_DECIDE, PETSC_DECIDE, nb_rows, nb_rows);
    MatSetType(y, MATMPIDENSE);
    MatSetFromOptions(y);
    MatSetUp(y);

    // Fill diagonal (each node fills a part)
    PetscInt istart, iend;
    MatGetOwnershipRange(x, &istart, &iend);
    PetscScalar value;
    for (PetscInt i = istart; i < iend; ++i)
    {
        MatGetValues(x, 1, &i, 1, &ZERO, &value);
        MatSetValues(y, 1, &i, 1, &i, &value, INSERT_VALUES);
    }

    MatAssemblyBegin(y, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(y, MAT_FINAL_ASSEMBLY);
    return y;
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
            while (num_previous_sample_indices < num_sample_indices && (sample_indices[num_previous_sample_indices]) <= (i - num_sample_indices + num_previous_sample_indices))
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

/*
Get the first n columns
*/
Mat GetFirstCols(Mat x, const unsigned int n)
{
    PetscInt nb_rows;
    MatGetSize(x, &nb_rows, NULL);

    Mat y;
    MatCreate(PETSC_COMM_WORLD, &y);
    MatSetSizes(y, PETSC_DECIDE, PETSC_DECIDE, nb_rows, n);
    MatSetType(y, MATMPIDENSE);
    MatSetFromOptions(y);
    MatSetUp(y);
    
    // Fill (each node a its part)
    PetscInt istart, iend;
    MatGetOwnershipRange(x, &istart, &iend);
    PetscInt *row_indices, *col_indices;
    row_indices = (PetscInt*) malloc(sizeof(PetscInt) * (iend-istart));
    col_indices = (PetscInt*) malloc(sizeof(PetscInt) * n);
    for (PetscInt i = 0; i < (iend-istart); ++i)
    {
        row_indices[i] = i + istart;
    }
    for (PetscInt i = 0; i < n; ++i)
    {
        col_indices[i] = i;
    }
    PetscScalar* values = (PetscScalar*) malloc(sizeof(PetscScalar) * (iend-istart) * n);

    MatGetValues(x, iend-istart, row_indices, n, col_indices, values);
    MatSetValues(y, iend-istart, row_indices, n, col_indices, values, INSERT_VALUES);

    free(values);
    free(col_indices);
    free(row_indices);

    MatAssemblyBegin(y, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(y, MAT_FINAL_ASSEMBLY);
    return y;
}

/*
Get the last n columns
*/
Mat GetLastCols(Mat x, const unsigned int n)
{
    PetscInt nb_rows, nb_cols;
    MatGetSize(x, &nb_rows, &nb_cols);

    Mat y;
    MatCreate(PETSC_COMM_WORLD, &y);
    MatSetSizes(y, PETSC_DECIDE, PETSC_DECIDE, nb_rows, n);
    MatSetType(y, MATMPIDENSE);
    MatSetFromOptions(y);
    MatSetUp(y);
    
    // Fill (each node a its part)
    PetscInt istart, iend;
    MatGetOwnershipRange(x, &istart, &iend);
    PetscInt *row_indices, *col_indices_x, *col_indices_y;
    row_indices = (PetscInt*) malloc(sizeof(PetscInt) * (iend-istart));
    col_indices_x = (PetscInt*) malloc(sizeof(PetscInt) * n);
    col_indices_y = (PetscInt*) malloc(sizeof(PetscInt) * n);
    for (PetscInt i = 0; i < (iend-istart); ++i)
    {
        row_indices[i] = i + istart;
    }
    for (PetscInt i = 0; i < n; ++i)
    {
        col_indices_x[i] = i + (nb_cols - n);
        col_indices_y[i] = i;
    }
    PetscScalar* values = (PetscScalar*) malloc(sizeof(PetscScalar) * (iend-istart) * n);

    MatGetValues(x, iend-istart, row_indices, n, col_indices_x, values);
    MatSetValues(y, iend-istart, row_indices, n, col_indices_y, values, INSERT_VALUES);

    free(values);
    free(col_indices_y);
    free(col_indices_x);
    free(row_indices);

    MatAssemblyBegin(y, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(y, MAT_FINAL_ASSEMBLY);
    return y;
}

/*
Get the first n rows
*/
Mat GetFirstRows(Mat x, const unsigned int n)
{
    PetscInt nb_rows, nb_cols;
    MatGetSize(x, &nb_rows, &nb_cols);

    Mat y;
    MatCreate(PETSC_COMM_WORLD, &y);
    MatSetSizes(y, PETSC_DECIDE, PETSC_DECIDE, n, nb_cols);
    MatSetType(y, MATMPIDENSE);
    MatSetFromOptions(y);
    MatSetUp(y);

    // Fill (each node a its part)
    PetscInt istart, iend;
    MatGetOwnershipRange(x, &istart, &iend);
    iend = min(iend, n);
    if (istart < iend)
    {
        PetscInt *row_indices, *col_indices;
        row_indices = (PetscInt*) malloc(sizeof(PetscInt) * (iend-istart));
        col_indices = (PetscInt*) malloc(sizeof(PetscInt) * nb_cols);
        for (PetscInt i = 0; i < (iend-istart); ++i)
        {
            row_indices[i] = i + istart;
        }
        for (PetscInt i = 0; i < nb_cols; ++i)
        {
            col_indices[i] = i;
        }
        PetscScalar* values = (PetscScalar*) malloc(sizeof(PetscScalar) * (iend-istart) * nb_cols);

        MatGetValues(x, iend-istart, row_indices, nb_cols, col_indices, values);
        MatSetValues(y, iend-istart, row_indices, nb_cols, col_indices, values, INSERT_VALUES);

        free(values);
        free(col_indices);
        free(row_indices);
    }

    MatAssemblyBegin(y, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(y, MAT_FINAL_ASSEMBLY);
    return y;
}

Mat SetNegativesToZero(Mat x)
{
    PetscInt nb_cols, nb_rows;
    MatType type;
    MatGetSize(x, &nb_rows, &nb_cols);
    MatGetType(x, &type);

    Mat y;
    MatCreate(PETSC_COMM_WORLD, &y);
    MatSetSizes(y, PETSC_DECIDE, PETSC_DECIDE, nb_rows, nb_cols);
    MatSetType(y, type);
    MatSetFromOptions(y);
    MatSetUp(y);

    // Fill
    PetscInt istart, iend;
    MatGetOwnershipRange(x, &istart, &iend);
    if (iend-istart > 0)
    {
        PetscInt *row_indices, *col_indices;
        row_indices = (PetscInt*) malloc(sizeof(PetscInt) * (iend-istart));
        col_indices = (PetscInt*) malloc(sizeof(PetscInt) * nb_cols);
        for (PetscInt i = 0; i < (iend-istart); ++i)
        {
            row_indices[i] = i + istart;
        }
        for (PetscInt i = 0; i < nb_cols; ++i)
        {
            col_indices[i] = i;
        }
        PetscScalar* values = (PetscScalar*) malloc(sizeof(PetscScalar) * (iend-istart) * nb_cols);

        MatGetValues(x, iend-istart, row_indices, nb_cols, col_indices, values);
        // Set to 0 when negative
        for (unsigned int i = 0; i < ((iend-istart) * nb_cols); ++i)
        {
            if (values[i] < 0.)
            {
                values[i] = 0.;
            }
        }
        MatSetValues(y, iend-istart, row_indices, nb_cols, col_indices, values, INSERT_VALUES);

        free(values);
        free(col_indices);
        free(row_indices);
    }

    MatAssemblyBegin(y, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(y, MAT_FINAL_ASSEMBLY);
    return y;
}

Vec MatRowSum(Mat A)
{
    PetscInt size;
    MatGetSize(A, &size, NULL);

    Vec D;
    VecCreate(PETSC_COMM_WORLD, &D);
    VecSetSizes(D, PETSC_DECIDE, size);
    VecSetFromOptions(D);

    MatGetRowSum(A, D);
    return D;
}

PetscScalar VecMean(Vec x)
{
    PetscScalar sum;
    // Sum
    VecSum(x, &sum);  // Collective

    // Divide
    PetscInt size;
    VecGetSize(x, &size);
    return sum / size;
}

/*
Returns identity matrix of rank n and with specified format
*/
Mat MatCreateIdentity(const unsigned int n, const MatType format)
{
    Mat ident;
    MatCreate(PETSC_COMM_WORLD, &ident);
    MatSetSizes(ident, PETSC_DECIDE, PETSC_DECIDE, n, n);
    MatSetType(ident, format);
    MatSetFromOptions(ident);
    MatSetUp(ident);

    Vec ones;
    VecCreate(PETSC_COMM_WORLD, &ones);
    VecSetSizes(ones, PETSC_DECIDE, n);
    VecSetFromOptions(ones);
    VecSet(ones, 1.0);
    MatDiagonalSet(ident, ones, INSERT_VALUES);
    VecDestroy(&ones);

    MatAssemblyBegin(ident, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(ident, MAT_FINAL_ASSEMBLY);
    return ident;
}

/*
vec_mat is a 1xN matrix with values between 0 and 1
Returns a filled png_bytes for proc with rank 0 only
Returns NULL for the others
*/
png_bytep* OneRowMat2pngbytes(Mat vec_mat, const unsigned int width, const unsigned int height, const int scale)
{
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    // Allocate memory for image for proc 0
    png_bytep* img_bytes = NULL;
    if (rank == 0)
    {
        img_bytes = (png_bytep*) malloc(sizeof(png_bytep) * height);
        for (unsigned int i = 0; i < height; ++i)
        {
            img_bytes[i] = (png_bytep) malloc(sizeof(png_byte) * width);
        }

        // All data is on process 0 already, get all values and cast to png_byte
        PetscInt* col_indices = (PetscInt*) malloc(sizeof(PetscInt) * width);
        PetscScalar* values = (PetscScalar*) malloc(sizeof(PetscScalar) * width);
        for (unsigned int i = 0; i < height; ++i)
        {
            for (unsigned int j = 0; j < width; ++j)
            {
                col_indices[j] = j + width*i;
            }
            MatGetValues(vec_mat, 1, &ZERO, width, col_indices, values);
            for (unsigned int j = 0; j < width; ++j)
            {
                img_bytes[i][j] = ((png_byte) (values[j] * scale));
            }
        }
        free(values);
        free(col_indices);
    }

    // Gather all data to process 0
    return img_bytes;
}

/*
All processes have the whole img_bytes
*/
Mat pngbytes2OneColMat(const png_bytep* const img_bytes, const unsigned int width, const unsigned int height)
{
    Mat x;
    MatCreate(PETSC_COMM_WORLD, &x);
    MatSetSizes(x, PETSC_DECIDE, PETSC_DECIDE, width*height, 1);
    MatSetType(x, MATMPIDENSE);
    MatSetFromOptions(x);
    MatSetUp(x);

    // Each process fills a part
    PetscInt istart, iend;
    PetscInt x_pos, y_pos;
    PetscScalar val;
    MatGetOwnershipRange(x, &istart, &iend);
    for (PetscInt i = istart; i < iend; ++i)
    {
        x_pos = num2x(i, width);
        y_pos = num2y(i, width);
        val = img_bytes[x_pos][y_pos];
        MatSetValues(x, 1, &i, 1, &ZERO, &val, INSERT_VALUES);
    }

    MatAssemblyBegin(x, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(x, MAT_FINAL_ASSEMBLY);
    return x;
}

/*
On process rank 0, return an allocated an filled image
The other processes get NULL
*/
png_bytep* OneColMat2pngbytes(Mat x, const unsigned int width, const unsigned int height)
{
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    Vec values, local_values;
    VecCreate(PETSC_COMM_WORLD, &values);
    VecSetSizes(values, PETSC_DECIDE, width*height);
    VecSetFromOptions(values);
    MatGetColumnVector(x, values, 0);
    VecScatter ctx;
    VecScatterCreateToZero(values, &ctx, &local_values);
    VecScatterBegin(ctx, values, local_values, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(ctx, values, local_values, INSERT_VALUES, SCATTER_FORWARD);

    // Allocate memory for image for proc 0
    png_bytep* img_bytes = NULL;
    if (rank == 0)
    {
        img_bytes = (png_bytep*) malloc(sizeof(png_bytep) * height);
        for (unsigned int i = 0; i < height; ++i)
        {
            img_bytes[i] = (png_bytep) malloc(sizeof(png_byte) * width);
        }

        // Fill img_bytes with local data and then receive the next data
        PetscInt x_pos, y_pos;
        PetscScalar val;
        for (PetscInt i = 0; i < (width*height); ++i)
        {
            x_pos = num2x(i, width);
            y_pos = num2y(i, width);
            VecGetValues(local_values, 1, &i, &val);
            img_bytes[x_pos][y_pos] = val;
        }
    }
    VecScatterDestroy(&ctx);
    VecDestroy(&local_values);
    VecDestroy(&values);

    // Gather all data to process 0
    return img_bytes;
}

Vec DiagMat2Vec(Mat x)
{
    PetscInt size;
    MatGetSize(x, &size, NULL);

    Vec v;
    VecCreate(PETSC_COMM_WORLD, &v);
    VecSetSizes(v, PETSC_DECIDE, size);
    VecSetFromOptions(v);

    MatGetDiagonal(x, v);

    return v;
}

void CopyVecs(Vec* in, Vec* out, const unsigned int p)
{
    for (unsigned int i = 0; i < p; ++i)
    {
        VecCopy(in[i], out[i]);
    }
}

Mat InverseDiagMat(Mat x)
{
    PetscInt n;
    MatGetSize(x, &n, NULL);

    Mat y;
    MatCreate(PETSC_COMM_WORLD, &y);
    MatSetSizes(y, PETSC_DECIDE, PETSC_DECIDE, n, n);
    MatSetType(y, MATMPIDENSE);
    MatSetFromOptions(y);
    MatSetUp(y);
    MatZeroEntries(y);

    // Each process fills a part of y
    PetscInt istart, iend;
    PetscScalar val;
    MatGetOwnershipRange(x, &istart, &iend);
    for (PetscInt i = istart; i < iend; ++i)
    {
        MatGetValues(x, 1, &i, 1, &i, &val);
        MatSetValue(y, i, i, 1./val, INSERT_VALUES);
    }

    MatAssemblyBegin(y, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(y, MAT_FINAL_ASSEMBLY);

    return y;
}

Vec pngbytes2Vec(const png_bytep* const img_bytes, const unsigned int width, const unsigned int height)
{
    const unsigned int size = width*height;

    Vec x;
    VecCreate(PETSC_COMM_WORLD, &x);
    VecSetSizes(x, PETSC_DECIDE, size);
    VecSetFromOptions(x);

    // Each process fills a part of vector
    PetscInt istart, iend;
    PetscInt x_pos, y_pos;
    VecGetOwnershipRange(x, &istart, &iend);
    for (PetscInt i = istart; i < iend; ++i)
    {
        x_pos = num2x(istart, width);
        y_pos = num2y(istart, width);
        VecSetValue(x, i, img_bytes[x_pos][y_pos], INSERT_VALUES);
    }

    VecAssemblyBegin(x);
    VecAssemblyEnd(x);
    return x;
}

png_bytep* Vec2pngbytes(Vec x, const unsigned int width, const unsigned int height)
{
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    Vec local_values;
    VecScatter ctx;
    VecScatterCreateToZero(x, &ctx, &local_values);
    VecScatterBegin(ctx, x, local_values, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(ctx, x, local_values, INSERT_VALUES, SCATTER_FORWARD);

    // Allocate memory for image for proc 0
    png_bytep* img_bytes = NULL;
    if (rank == 0)
    {
        img_bytes = (png_bytep*) malloc(sizeof(png_bytep) * height);
        for (unsigned int i = 0; i < height; ++i)
        {
            img_bytes[i] = (png_bytep) malloc(sizeof(png_byte) * width);
        }

        // Fill img_bytes with local data and then receive the next data
        PetscInt x_pos, y_pos;
        PetscScalar val;
        for (PetscInt i = 0; i < (width*height); ++i)
        {
            x_pos = num2x(i, width);
            y_pos = num2y(i, width);
            VecGetValues(local_values, 1, &i, &val);
            img_bytes[x_pos][y_pos] = val;
        }
    }
    VecScatterDestroy(&ctx);
    VecDestroy(&local_values);

    // Gather all data to process 0
    return img_bytes;
}

Mat AboveXSetY(Mat x, PetscScalar X, PetscScalar Y)
{
    PetscInt nb_cols, nb_rows;
    MatType type;
    MatGetSize(x, &nb_rows, &nb_cols);
    MatGetType(x, &type);

    Mat y;
    MatCreate(PETSC_COMM_WORLD, &y);
    MatSetSizes(y, PETSC_DECIDE, PETSC_DECIDE, nb_rows, nb_cols);
    MatSetType(y, type);
    MatSetFromOptions(y);
    MatSetUp(y);

    // Fill
    PetscInt istart, iend;
    MatGetOwnershipRange(x, &istart, &iend);
    if (iend-istart > 0)
    {
        PetscInt *row_indices, *col_indices;
        row_indices = (PetscInt*) malloc(sizeof(PetscInt) * (iend-istart));
        col_indices = (PetscInt*) malloc(sizeof(PetscInt) * nb_cols);
        for (PetscInt i = 0; i < (iend-istart); ++i)
        {
            row_indices[i] = i + istart;
        }
        for (PetscInt i = 0; i < nb_cols; ++i)
        {
            col_indices[i] = i;
        }
        PetscScalar* values = (PetscScalar*) malloc(sizeof(PetscScalar) * (iend-istart) * nb_cols);

        MatGetValues(x, iend-istart, row_indices, nb_cols, col_indices, values);
        // Set to 0 when negative
        for (unsigned int i = 0; i < ((iend-istart) * nb_cols); ++i)
        {
            if (values[i] > X)
            {
                values[i] = Y;
            }
        }
        MatSetValues(y, iend-istart, row_indices, nb_cols, col_indices, values, INSERT_VALUES);

        free(values);
        free(col_indices);
        free(row_indices);
    }

    MatAssemblyBegin(y, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(y, MAT_FINAL_ASSEMBLY);
    return y;
}
