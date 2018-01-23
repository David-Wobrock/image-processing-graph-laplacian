#include "affinity.h"

#include <stdlib.h>
#include <petscvec.h>

#include "utils.h"

static void ComputePhotometricDistance(Vec sample_pixels_value, const double sample_value, const double h, Vec v)
{
    VecCopy(sample_pixels_value, v);
    // Operation exp(-abs(pow((double) (sample_value - all), 2))/pow(h, 2));
    VecShift(v, -sample_value);
    VecAbs(v);
    VecPow(v, 2);
    VecScale(v, -(1./(h*h)));
    VecExp(v);
}

static void ComputeSpatialFilter(Vec sample_pixels_x, Vec sample_pixels_y, const unsigned int sample_x, const unsigned int sample_y, const double h, Vec v)
{
    PetscInt sample_size;
    VecGetSize(sample_pixels_x, &sample_size);

    // Compute location part exp(-norm(x_i - x_j)^2 / h^2)
    Vec x_loc, y_loc;
    VecDuplicate(v, &x_loc);
    VecCopy(sample_pixels_x, x_loc);
    VecDuplicate(v, &y_loc);
    VecCopy(sample_pixels_y, y_loc);

    VecShift(x_loc, -((PetscScalar) sample_x));
    VecPow(x_loc, 2);
    VecShift(y_loc, -((PetscScalar) sample_y));
    VecPow(y_loc, 2);
    PetscInt* col_indices = (PetscInt*) malloc(sizeof(PetscInt) * sample_size);
    for (unsigned int i = 0; i < sample_size; ++i)
    {
        col_indices[i] = i;
    }
    PetscScalar* x_values = (PetscScalar*) malloc(sizeof(PetscScalar) * sample_size);
    PetscScalar* y_values = (PetscScalar*) malloc(sizeof(PetscScalar) * sample_size);
    VecGetValues(x_loc, sample_size, col_indices, x_values);
    VecGetValues(y_loc, sample_size, col_indices, y_values);
    for (unsigned int i = 0; i < sample_size; ++i)
    {
        x_values[i] = x_values[i] + y_values[i];
    }
    VecSetValues(v, sample_size, col_indices, x_values, INSERT_VALUES);
    free(x_values);
    free(y_values);
    free(col_indices);
    VecDestroy(&y_loc);
    VecDestroy(&x_loc);

    VecScale(v, -(1./(h*h)));
    VecExp(v);
}

static void ComputeBilateralFilter(Vec sample_pixels_x, Vec sample_pixels_y, Vec sample_pixels_value, const unsigned int sample_x, const unsigned int sample_y, const double sample_value, const double h_loc, const double h_val, Vec v)
{
    Vec loc_part, val_part;
    VecDuplicate(v, &loc_part);
    VecDuplicate(v, &val_part);

    // Compute location part exp(-norm(x_i - x_j)^2 / h^2)
    Vec x_loc, y_loc;
    VecDuplicate(v, &x_loc);
    VecCopy(sample_pixels_x, x_loc);
    VecDuplicate(v, &y_loc);
    VecCopy(sample_pixels_y, y_loc);

    VecShift(x_loc, -((PetscScalar) sample_x));
    VecPow(x_loc, 2);
    VecShift(y_loc, -((PetscScalar) sample_y));
    VecPow(y_loc, 2);
    PetscInt sample_size;
    VecGetSize(x_loc, &sample_size);
    PetscInt* col_indices = (PetscInt*) malloc(sizeof(PetscInt) * sample_size);
    for (unsigned int i = 0; i < sample_size; ++i)
    {
        col_indices[i] = i;
    }
    PetscScalar* x_values = (PetscScalar*) malloc(sizeof(PetscScalar) * sample_size);
    PetscScalar* y_values = (PetscScalar*) malloc(sizeof(PetscScalar) * sample_size);
    VecGetValues(x_loc, sample_size, col_indices, x_values);
    VecGetValues(y_loc, sample_size, col_indices, y_values);
    for (unsigned int i = 0; i < sample_size; ++i)
    {
        x_values[i] = x_values[i] + y_values[i];
    }
    VecSetValues(loc_part, sample_size, col_indices, x_values, INSERT_VALUES);
    free(x_values);
    free(y_values);
    free(col_indices);
    VecDestroy(&y_loc);
    VecDestroy(&x_loc);

    VecScale(loc_part, -(1./(h_loc*h_loc)));
    VecExp(loc_part);

    // Compute value part exp(-(y_i - y_j)^2 / h^2)
    VecCopy(sample_pixels_value, val_part);
    VecShift(val_part, -sample_value);
    VecAbs(val_part);
    VecPow(val_part, 2);
    VecScale(val_part, -(1./(h_val*h_val)));
    VecExp(val_part);

    // Multiply both together
    VecPointwiseMult(v, loc_part, val_part);
    VecDestroy(&loc_part);
    VecDestroy(&val_part);
}

void ComputeDistance(Vec sample_pixels_x, Vec sample_pixels_y, Vec sample_pixels_value, const unsigned int sample_x, const unsigned int sample_y, const double sample_value, Vec v)
{
    const PetscScalar h_val = 20.;
    const PetscScalar h_loc = 40.;
    //ComputePhotometricDistance(sample_pixels_value, sample_value, h_val, v);
    //ComputeSpatialFilter(sample_pixels_x, sample_pixels_y, sample_x, sample_y, h_loc, v);
    ComputeBilateralFilter(sample_pixels_x, sample_pixels_y, sample_pixels_value, sample_x, sample_y, sample_value, h_loc, h_val, v);
}

/*
Compute the affinity matrices K_A and K_B
Input: Created variables K_A and K_B, images and its size, the number of samples and the samples indices
Output: created and setup K_A and K_B matrices
*/
void ComputeAffinityMatrices(Mat* K_A, Mat* K_B, const png_bytep* const img_bytes, const int width, const int height, const unsigned int sample_size, const unsigned int* sample_indices)
{
    int num_pixels = width * height;
    PetscInt istart, iend;
    PetscScalar* values;
    PetscInt* col_indices;
    Vec v, sample_pixels_value, sample_pixels_x, sample_pixels_y;

    // K_A
    MatCreate(PETSC_COMM_WORLD, K_A);
    MatSetSizes(*K_A, PETSC_DECIDE, PETSC_DECIDE, sample_size, sample_size);
    MatSetType(*K_A, MATMPIDENSE);
    MatSetFromOptions(*K_A);
    MatSetUp(*K_A);

    values = (PetscScalar*) malloc(sizeof(PetscScalar) * sample_size); // One row at a time
    col_indices = (PetscInt*) malloc(sizeof(PetscInt) * sample_size);
    double sample_value;
    for (unsigned int i = 0; i < sample_size; ++i)
    {
        col_indices[i] = i;
    }

    VecCreateSeq(PETSC_COMM_SELF, sample_size, &sample_pixels_x);
    VecCreateSeq(PETSC_COMM_SELF, sample_size, &sample_pixels_y);
    VecCreateSeq(PETSC_COMM_SELF, sample_size, &sample_pixels_value);
    unsigned int sample_idx, sample_x, sample_y;
    // These vectors will contain all gray values, x and y of the sample pixels
    // Each line is computed locally, so they are local
    for (unsigned int j = 0; j < sample_size; ++j)
    {
        sample_idx = sample_indices[j];
        sample_x = num2x(sample_idx, width);
        sample_y = num2y(sample_idx, width);
        VecSetValue(
            sample_pixels_x,
            j,
            sample_x,
            INSERT_VALUES);
        VecSetValue(
            sample_pixels_y,
            j,
            sample_y,
            INSERT_VALUES);
        VecSetValue(
            sample_pixels_value,
            j,
            img_bytes[sample_x][sample_y],
            INSERT_VALUES);
    }
    VecDuplicate(sample_pixels_value, &v);

    MatGetOwnershipRange(*K_A, &istart, &iend);
    for (unsigned int i = istart; i < iend; ++i)
    {
        sample_idx = sample_indices[i];
        sample_x = num2x(sample_idx, width);
        sample_y = num2y(sample_idx, width);
        sample_value = img_bytes[sample_x][sample_y];

        ComputeDistance(sample_pixels_x, sample_pixels_y, sample_pixels_value, sample_x, sample_y, sample_value, v);

        VecGetValues(v, sample_size, col_indices, values);
        MatSetValues(*K_A, 1, (int*)&i, sample_size, col_indices, values, ADD_VALUES);
    }
    VecDestroy(&v);
    VecDestroy(&sample_pixels_value);
    VecDestroy(&sample_pixels_y);
    VecDestroy(&sample_pixels_x);
    free(col_indices);
    free(values);

    // K_B
    unsigned int remaining_pixels_size = num_pixels - sample_size;
    MatCreate(PETSC_COMM_WORLD, K_B);
    MatSetSizes(*K_B, PETSC_DECIDE, PETSC_DECIDE, sample_size, remaining_pixels_size);
    MatSetType(*K_B, MATMPIDENSE);
    MatSetFromOptions(*K_B);
    MatSetUp(*K_B);

    values = (PetscScalar*) malloc(sizeof(PetscScalar) * remaining_pixels_size); // One row at a time
    col_indices = (PetscInt*) malloc(sizeof(PetscInt) * remaining_pixels_size);
    for (unsigned int i = 0; i < remaining_pixels_size; ++i)
    {
        col_indices[i] = i;
    }
    VecCreateSeq(PETSC_COMM_SELF, remaining_pixels_size, &sample_pixels_value);
    VecDuplicate(sample_pixels_value, &sample_pixels_x);
    VecDuplicate(sample_pixels_value, &sample_pixels_y);
    unsigned int tmp_idx = 0;
    unsigned int idx = 0;
    for (unsigned int j = 0; j < num_pixels; ++j)
    {
        if (j != sample_indices[tmp_idx])
        {
            sample_x = num2x(j, width);
            sample_y = num2y(j, width);
            VecSetValue(sample_pixels_x, idx, sample_x, INSERT_VALUES);
            VecSetValue(sample_pixels_y, idx, sample_y, INSERT_VALUES);
            VecSetValue(sample_pixels_value, idx, img_bytes[sample_x][sample_y], INSERT_VALUES);
            ++idx;
        }
        else
        {
            ++tmp_idx;
        }
    }

    VecDuplicate(sample_pixels_value, &v);
    MatGetOwnershipRange(*K_B, &istart, &iend);
    for (unsigned int i = istart; i < iend; ++i)
    {
        sample_idx = sample_indices[i];
        sample_x = num2x(sample_idx, width);
        sample_y = num2y(sample_idx, width);
        sample_value = img_bytes[sample_x][sample_y];

        ComputeDistance(sample_pixels_x, sample_pixels_y, sample_pixels_value, sample_x, sample_y, sample_value, v);

        VecGetValues(v, remaining_pixels_size, col_indices, values);
        MatSetValues(*K_B, 1, (int*)&i, remaining_pixels_size, col_indices, values, INSERT_VALUES);
    }
    VecDestroy(&sample_pixels_value);
    VecDestroy(&sample_pixels_y);
    VecDestroy(&sample_pixels_x);
    VecDestroy(&v);
    free(col_indices);
    free(values);

    MatAssemblyBegin(*K_A, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(*K_B, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*K_A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*K_B, MAT_FINAL_ASSEMBLY);
}
