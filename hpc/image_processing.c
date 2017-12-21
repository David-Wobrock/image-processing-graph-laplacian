#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <mpi.h>
#include <png.h>

#include <petscsys.h>
#include <petscmat.h>

#include <slepcsys.h>
#include <slepceps.h>

#include "read_img.h"
#include "utils.h"
#include "inverse_power_it.h"

/*
Initialize Slepc/Petsc/MPI
Input: argc, argv, the parameters of main()
Output: rank and size
*/
static PetscErrorCode InitProgram(int argc, char** argv, int* const rank, int* const size) 
{
    PetscErrorCode ierr = 0;

    ierr = SlepcInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);
    MPI_Comm_rank(PETSC_COMM_WORLD, rank);
    MPI_Comm_size(PETSC_COMM_WORLD, size);
    return ierr;
}

/*
Input: an allocate space for filename
Output: filled filename
*/
static PetscErrorCode GetFilePath(char* const filename)
{
    PetscErrorCode ierr = 0;
    PetscBool found_filename;
    
    ierr = PetscOptionsGetString(NULL, NULL, "-f", filename, PETSC_MAX_PATH_LEN, &found_filename); CHKERRQ(ierr);
    if (!found_filename) {
        fprintf(stderr, "No filename found (option -f)\n");
        exit(1);
    }
    return ierr;
}

/*
Read image and broadcast it to everyone
Input: rank, filename, *not* allocated img_bytes
Output: allocated and filled img_bytes, filled image width and height
*/
static void ReadAndBcastImage(const int rank, const char* const filename, png_bytep** const img_bytes, int* const width, int* const height)
{
    int width_height[2];
    if (rank == 0)
    {
        // Read image
        read_png(filename, img_bytes, width, height);
        // Broadcast image size
        width_height[0] = *width;
        width_height[1] = *height;
        MPI_Bcast(width_height, 2, MPI_INT, 0, PETSC_COMM_WORLD);
        // Broadcast image
        for (unsigned int i = 0; i < *height; ++i)
        {
            MPI_Bcast((*img_bytes)[i], *width, MPI_CHAR, 0, PETSC_COMM_WORLD);
        }
    }
    else
    {
        // Receive image size
        MPI_Bcast(width_height, 2, MPI_INT, 0, PETSC_COMM_WORLD);
        *width = width_height[0];
        *height = width_height[1];
        // Alloc and receive image
        *img_bytes = (png_bytep*) malloc(sizeof(png_bytep) * (*height));
        for (unsigned int i = 0; i < *height; ++i)
        {
            (*img_bytes)[i] = (png_bytep) malloc(sizeof(png_byte) * (*width));
            MPI_Bcast((*img_bytes)[i], *width, MPI_CHAR, 0, PETSC_COMM_WORLD);
        }
    }
}

/*
Get indices of the sampled pixels (from 0 to width*height)
Input: image width and height, the number of requested samples (as pointer because it will be modified), a *not* allocated pointer for the indices
Output: the exact number of samples (sample_size) and a filled and allocated array with the indices (sample_indices)
*/
static void Sampling(const int width, const int height, unsigned int* const sample_size, unsigned int** const sample_indices)
{
    const unsigned int sample_dist = (unsigned int) (sqrt((width*height) / (*sample_size)));
    const unsigned int xy0 = (unsigned int) (sample_dist/2);
    const unsigned int size_x_span = (unsigned int) ceil((height - 1 - xy0) / sample_dist) + 1;
    const unsigned int size_y_span = (unsigned int) ceil((width - 1 - xy0) / sample_dist) + 1;

    *sample_size = size_x_span * size_y_span;
    *sample_indices = (unsigned int*) malloc(sizeof(unsigned int) * (*sample_size));
    unsigned int c = 0;
    for (unsigned int i = xy0; i < height-1; i += sample_dist)
        for (unsigned int j = xy0; j < width-1; j += sample_dist)
            (*sample_indices)[c++] = width*i + j;
}

/*
Photometric distance
*/
static void ComputeDistance(Vec v, const double sample_value)
{
    PetscScalar h = 10.;

    // Operation exp(-abs(pow((double) (sample_value - all), 2))/pow(h, 2));
    VecShift(v, -sample_value);
    VecPow(v, 2);
    VecAbs(v);
    VecScale(v, -(1./(h*h)));
    VecExp(v);
}

/*
Compute the affinity matrices K_A and K_B
Input: Created variables K_A and K_B, images and its size, the number of samples and the samples indices
Output: created and setup K_A and K_B matrices
*/
static void ComputeAffinityMatrices(Mat* K_A, Mat* K_B, const png_bytep* const img_bytes, const int width, const int height, const unsigned int sample_size, const unsigned int* sample_indices)
{
    int num_pixels = width * height;
    PetscInt istart, iend;
    PetscScalar* values;
    PetscInt* col_indices;
    Vec v, v_orig;

    // K_A
    //MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, sample_size, sample_size, NULL, K_A);
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
    for (unsigned int j = 0; j < sample_size; ++j)
        VecSetValue(v_orig, j, img_bytes[num2x(sample_indices[j], width)][num2y(sample_indices[j], width)], INSERT_VALUES);
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
    //MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, sample_size, remaining_pixels_size, NULL, K_B);
    MatCreate(PETSC_COMM_WORLD, K_B);
    MatSetSizes(*K_B, PETSC_DECIDE, PETSC_DECIDE, sample_size, remaining_pixels_size);
    MatSetType(*K_B, MATMPIDENSE);
    MatSetFromOptions(*K_B);
    MatSetUp(*K_B);

    values = (PetscScalar*) malloc(sizeof(PetscScalar) * remaining_pixels_size); // One row at a time
    col_indices = (PetscInt*) malloc(sizeof(PetscInt) * remaining_pixels_size);
    for (unsigned int i = 0; i < remaining_pixels_size; ++i)
        col_indices[i] = i;
    VecCreateSeq(PETSC_COMM_SELF, remaining_pixels_size, &v_orig);
    unsigned int tmp_idx = 0;
    unsigned int idx = 0;
    for (unsigned int j = 0; j < num_pixels; ++j)
    {
        if (j != sample_indices[tmp_idx])
            VecSetValue(v_orig, idx++, img_bytes[num2x(j, width)][num2y(j, width)], INSERT_VALUES);
        else
            ++tmp_idx;
    }
    VecDuplicate(v_orig, &v);
    MatGetOwnershipRange(*K_B, &istart, &iend);
    for (unsigned int i = istart; i < iend; ++i)
    {
        sample_index = sample_indices[i];
        sample_value = img_bytes[num2x(sample_index, width)][num2y(sample_index, width)];

        VecCopy(v_orig, v); // Start from original pixel values
        ComputeDistance(v, sample_value);

        VecGetValues(v, sample_size, col_indices, values);
        MatSetValues(*K_B, 1, (int*)&i, remaining_pixels_size, col_indices, values, ADD_VALUES);
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

int main(int argc, char** argv)
{
    char filename[PETSC_MAX_PATH_LEN];
    PetscMPIInt rank, size;
    Mat K_A, K_B;

    InitProgram(argc, argv, &rank, &size);
    double start_time = MPI_Wtime();
    GetFilePath(filename);

    int width, height;
    png_bytep* img_bytes;
    ReadAndBcastImage(rank, filename, &img_bytes, &width, &height);
    //printf("I am %d of %d and width=%d, height=%d and first val=%d\n", rank, size, width, height, img_bytes[0][0]);

    // Sampling (all compute the same locally)
    unsigned int sample_size = width*height*0.01; // 1%
    unsigned int* sample_indices;
    Sampling(width, height, &sample_size, &sample_indices);
    PetscPrintf(PETSC_COMM_WORLD, "Sample size: %d\n", sample_size);

    // Compute affinity matrix
    double start_affinity = MPI_Wtime();
    ComputeAffinityMatrices(&K_A, &K_B, img_bytes, width, height, sample_size, sample_indices);
    PetscPrintf(PETSC_COMM_WORLD, "Affinity matrices computation time: %fs\n", MPI_Wtime() - start_affinity);

    // Solve eigenvalue problem
    const unsigned int p = sample_size-1; // num eigenpairs
    PetscReal* eigenvalues = InversePowerIteration(K_A, sample_size, p);
    for (unsigned int i = 0; i < p; ++i)
    {
        PetscPrintf(PETSC_COMM_WORLD, "%f /", eigenvalues[i]);
    }
    PetscPrintf(PETSC_COMM_WORLD, "\n");
    free(eigenvalues);

    // SLEPc eigenvalues
    EPS eps;
    EPSCreate(PETSC_COMM_WORLD, &eps);
    EPSSetOperators(eps, K_A, NULL);
    EPSSetProblemType(eps, EPS_HEP);
    //EPSSetWhichEigenpairs(eps, EPS_SMALLEST_MAGNITUDE);
    EPSSetDimensions(eps, p, PETSC_DEFAULT, PETSC_DEFAULT);
    EPSSetFromOptions(eps);

    double start_solve = MPI_Wtime();
    EPSSolve(eps);
    PetscPrintf(PETSC_COMM_WORLD, "Solve time for SLEPc EPS: %fs\n", MPI_Wtime() - start_solve);

    EPSValuesView(eps, PETSC_VIEWER_STDOUT_WORLD);

    EPSDestroy(&eps);

    // Nyström
    // TODO

    // End
    PetscPrintf(PETSC_COMM_WORLD, "Total computation time: %fs\n", MPI_Wtime() - start_time);

    // Clean up
    MatDestroy(&K_A);
    MatDestroy(&K_B);
    free(sample_indices);

    for (unsigned int i = 0; i < height; ++i)
        free(img_bytes[i]);
    free(img_bytes);

    SlepcFinalize();

    return 0;
}
