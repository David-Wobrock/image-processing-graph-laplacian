#include "affinity.h"

#include <stdlib.h>
#include <petscvec.h>

#include "utils.h"
#include "write_img.h"

/*
Photometric distance
*/
static void ComputePhotometricDistance(Vec v, const double sample_value, const double h)
{
    // Operation exp(-abs(pow((double) (sample_value - all), 2))/pow(h, 2));
    VecShift(v, -sample_value);
    VecPow(v, 2);
    VecAbs(v);
    VecScale(v, -(1./(h*h)));
    VecExp(v);
}

static void ComputeDistance(Vec v, const double sample_value)
{
    const PetscScalar h = 10.;
    ComputePhotometricDistance(v, sample_value, h);
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
    Vec v, v_orig;

    // K_A
    MatCreate(PETSC_COMM_WORLD, K_A);
    MatSetSizes(*K_A, PETSC_DECIDE, PETSC_DECIDE, sample_size, sample_size);
    MatSetType(*K_A, MATMPIDENSE);
    MatSetFromOptions(*K_A);
    MatSetUp(*K_A);

    values = (PetscScalar*) malloc(sizeof(PetscScalar) * sample_size); // One row at a time
    col_indices = (PetscInt*) malloc(sizeof(PetscInt) * sample_size);
    unsigned int sample_index;
    double sample_value;
    for (unsigned int i = 0; i < sample_size; ++i)
    {
        col_indices[i] = i;
    }

    VecCreateSeq(PETSC_COMM_SELF, sample_size, &v_orig);
    unsigned int sample_idx;
    for (unsigned int j = 0; j < sample_size; ++j)
    {
        sample_idx = sample_indices[j];
        VecSetValue(
            v_orig,
            j,
            img_bytes[num2x(sample_idx, width)][num2y(sample_idx, width)],
            INSERT_VALUES);
    }
    VecDuplicate(v_orig, &v);

    MatGetOwnershipRange(*K_A, &istart, &iend);
    for (unsigned int i = istart; i < iend; ++i)
    {
        sample_index = sample_indices[i];
        sample_value = img_bytes[num2x(sample_index, width)][num2y(sample_index, width)];

        VecCopy(v_orig, v); // Start from original pixel values
        ComputeDistance(v, sample_value);

        VecGetValues(v, sample_size, col_indices, values);
        MatSetValues(*K_A, 1, (int*)&i, sample_size, col_indices, values, ADD_VALUES);
    }
    VecDestroy(&v_orig);
    VecDestroy(&v);
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
    VecCreateSeq(PETSC_COMM_SELF, remaining_pixels_size, &v_orig);
    unsigned int tmp_idx = 0;
    unsigned int idx = 0;
    for (unsigned int j = 0; j < num_pixels; ++j)
    {
        if (j != sample_indices[tmp_idx])
        {
            VecSetValue(v_orig, idx++, img_bytes[num2x(j, width)][num2y(j, width)], INSERT_VALUES);
        }
        else
        {
            ++tmp_idx;
        }
    }

    VecDuplicate(v_orig, &v);
    MatGetOwnershipRange(*K_B, &istart, &iend);
    for (unsigned int i = istart; i < iend; ++i)
    {
        sample_index = sample_indices[i];
        sample_value = img_bytes[num2x(sample_index, width)][num2y(sample_index, width)];

        VecCopy(v_orig, v); // Start from original pixel values
        ComputeDistance(v, sample_value);

        VecGetValues(v, remaining_pixels_size, col_indices, values);
        MatSetValues(*K_B, 1, (int*)&i, remaining_pixels_size, col_indices, values, INSERT_VALUES);
    }
    VecDestroy(&v_orig);
    VecDestroy(&v);
    free(col_indices);
    free(values);

    MatAssemblyBegin(*K_A, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(*K_B, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*K_A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*K_B, MAT_FINAL_ASSEMBLY);
}

static void ComputeAndSaveAffinityMatrixOfPixelNum(Mat phi, Mat Pi, const unsigned int width, const unsigned int height, const unsigned int pixel_num)
{
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    PetscInt p;
    MatGetSize(phi, NULL, &p);

    // Get one row of the eigenvalues (as diagonal matrix)
    Mat phi_Vec;
    MatCreate(PETSC_COMM_WORLD, &phi_Vec);
    MatSetSizes(phi_Vec, PETSC_DECIDE, PETSC_DECIDE, 1, p);
    MatSetType(phi_Vec, MATMPIDENSE);
    MatSetFromOptions(phi_Vec);
    MatSetUp(phi_Vec);
    // Fill the diagonal (only the proc possessing the row)
    PetscInt start, end;
    MatGetOwnershipRange(phi, &start, &end);
    if (start <= pixel_num && pixel_num < end) {
        const PetscScalar* values;
        MatGetRow(phi, pixel_num, NULL, NULL, &values);
        const int zero = 0;
        for (unsigned int i = 0; i < p; ++i)
        {
            MatSetValues(phi_Vec, 1, &zero, 1, (int*)&i, values+i, INSERT_VALUES);
        }
        MatRestoreRow(phi, pixel_num, NULL, NULL, &values);
    }
    MatAssemblyBegin(phi_Vec, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(phi_Vec, MAT_FINAL_ASSEMBLY);

    // Mult phi_Vec and Pi
    Mat tmp;
    MatMatMult(phi_Vec, Pi, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &tmp);
    MatDestroy(&phi_Vec);

    // Mult result with phi_T
    Mat phi_T, affinity_img_on_vec;
    //MatMatTransposeMult(tmp, phi, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &affinity_img_on_vec);
    MatTranspose(phi, MAT_INITIAL_MATRIX, &phi_T);
    MatMatMult(tmp, phi_T, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &affinity_img_on_vec);
    MatDestroy(&phi_T);
    MatDestroy(&tmp);

    // Rearrange vector into image (on one proc) and save
    png_bytep* img_bytes = VecMat2pngbytes(affinity_img_on_vec, width, height, 255);
    MatDestroy(&affinity_img_on_vec);
    if (rank == 0)
    {
        write_png("affinity.png", img_bytes, width, height);
        for (unsigned int i = 0; i < height; ++i)
        {
            free(img_bytes[i]);
        }
        free(img_bytes);
    }
}

void ComputeAndSaveAffinityMatrixOfPixel(Mat phi, Mat Pi, const unsigned int width, const unsigned int height, const unsigned int pixel_x, const unsigned int pixel_y)
{
    ComputeAndSaveAffinityMatrixOfPixelNum(phi, Pi, width, height, xy2num(pixel_x, pixel_y, width));
}
