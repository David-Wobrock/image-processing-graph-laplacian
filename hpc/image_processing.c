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
#include "write_img.h"
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
    const unsigned int size_x_span = (unsigned int) ceil((height - 1 - xy0) / (double) sample_dist);
    const unsigned int size_y_span = (unsigned int) ceil((width - 1 - xy0) / (double) sample_dist);

    *sample_size = size_x_span * size_y_span;
    *sample_indices = (unsigned int*) malloc(sizeof(unsigned int) * (*sample_size));
    unsigned int c = 0;
    for (unsigned int i = xy0; i < height-1; i += sample_dist)
    {
        for (unsigned int j = xy0; j < width-1; j += sample_dist)
        {
            (*sample_indices)[c++] = width*i + j;
        }
    }
}

/*
Photometric distance
*/
static void ComputeDistance(Vec v, const double sample_value)
{
    const PetscScalar h = 10.;

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


static Mat Nystroem(Mat B, Mat Phi_A, Mat Pi_A_Inv, const unsigned int N, const unsigned n, const unsigned p)
{
    Mat phi;
    MatCreate(PETSC_COMM_WORLD, &phi);
    MatSetSizes(phi, PETSC_DECIDE, PETSC_DECIDE, N, p);
    MatSetType(phi, MATMPIDENSE);
    MatSetFromOptions(phi);
    MatSetUp(phi);

    // Fill upper part (each node fills a part of the matrix)
    PetscInt istart, iend;
    PetscInt *col_indices, *row_indices;
    PetscScalar *values;

    col_indices = (PetscInt*) malloc(sizeof(PetscInt) * p);
    for (unsigned int i = 0; i < p; ++i)
    {
        col_indices[i] = i;
    }

    MatGetOwnershipRange(Phi_A, &istart, &iend);
    values = (PetscScalar*) malloc(sizeof(PetscScalar) * p * (iend-istart));
    row_indices = (PetscInt*) malloc(sizeof(PetscInt) * (iend-istart));
    for (unsigned int i = 0; i < (iend-istart); ++i)
    {
        row_indices[i] = istart+i;
    }

    MatGetValues(Phi_A, iend-istart, row_indices, p, col_indices, values);
    MatSetValues(phi, iend-istart, row_indices, p, col_indices, values, INSERT_VALUES);

    free(values);
    free(row_indices);

    // Fill lower part (each node fills a part of the matrix)
    Mat lower, part_lower;
    MatTransposeMatMult(B, Phi_A, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &part_lower);
    MatMatMult(part_lower, Pi_A_Inv, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &lower);
    MatDestroy(&part_lower);

    MatGetOwnershipRange(lower, &istart, &iend);
    values = (PetscScalar*) malloc(sizeof(PetscScalar) * p * (iend-istart));
    row_indices = (PetscInt*) malloc(sizeof(PetscInt) * (iend-istart));
    for (unsigned int i = 0; i < (iend-istart); ++i)
    {
        row_indices[i] = istart+i;
    }
    MatGetValues(lower, iend-istart, row_indices, p, col_indices, values);
    for (unsigned int i = 0; i < (iend-istart); ++i)
    {
        row_indices[i] = istart+i+n;
    }
    MatSetValues(phi, iend-istart, row_indices, p, col_indices, values, INSERT_VALUES);
    MatDestroy(&lower);

    free(values);
    free(row_indices);
    free(col_indices);

    // Assemble matrix
    MatAssemblyBegin(phi, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(phi, MAT_FINAL_ASSEMBLY);

    return phi;
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
    MatView(phi_Vec, PETSC_VIEWER_STDOUT_WORLD);
    MatView(Pi, PETSC_VIEWER_STDOUT_WORLD);
    MatView(tmp, PETSC_VIEWER_STDOUT_WORLD);
    MatDestroy(&phi_Vec);

    // Mult result with phi_T
    Mat phi_T, affinity_img_on_vec;
    //MatMatTransposeMult(tmp, phi, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &affinity_img_on_vec);
    MatTranspose(phi, MAT_INITIAL_MATRIX, &phi_T);
    MatMatMult(tmp, phi_T, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &affinity_img_on_vec);
    MatDestroy(&phi_T);
    MatDestroy(&tmp);

    MatView(affinity_img_on_vec, PETSC_VIEWER_STDOUT_WORLD);
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

static void ComputeAndSaveAffinityMatrixOfPixel(Mat phi, Mat Pi, const unsigned int width, const unsigned int height, const unsigned int pixel_x, const unsigned int pixel_y)
{
    ComputeAndSaveAffinityMatrixOfPixelNum(phi, Pi, width, height, xy2num(pixel_x, pixel_y, width));
}

int main(int argc, char** argv)
{
    char filename[PETSC_MAX_PATH_LEN];
    PetscMPIInt rank, size;

    InitProgram(argc, argv, &rank, &size);
    double start_time = MPI_Wtime();
    GetFilePath(filename);

    int width, height;
    png_bytep* img_bytes;
    ReadAndBcastImage(rank, filename, &img_bytes, &width, &height);
    PetscPrintf(PETSC_COMM_WORLD, "Read image %s of size %dx%d\n", filename, width, height);

    // Sampling (all compute the same locally)
    unsigned int sample_size = width*height*0.01; // 1%
    //unsigned int sample_size = 50;
    unsigned int* sample_indices; // Must be sorted ASC
    Sampling(width, height, &sample_size, &sample_indices);
    PetscPrintf(PETSC_COMM_WORLD, "Sample size: %d\n", sample_size);

    // Compute affinity matrix
    double start_affinity = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD, "Computing affinity matrices... ");
    Mat K_A, K_B;
    ComputeAffinityMatrices(&K_A, &K_B, img_bytes, width, height, sample_size, sample_indices);
    PetscPrintf(PETSC_COMM_WORLD, "%fs\n", MPI_Wtime() - start_affinity);

    // Solve eigenvalue problem
    // Use SLEPc because we need greatest eigenelements
    double start_eps = MPI_Wtime();
    const unsigned int p = sample_size; // num eigenpairs
    PetscPrintf(PETSC_COMM_WORLD, "Computing %d largest eigenvalues of affinity matrix... ", p);
    EPS eps;
    EPSCreate(PETSC_COMM_WORLD, &eps);
    EPSSetOperators(eps, K_A, NULL);
    EPSSetProblemType(eps, EPS_HEP); // Symmetric
    EPSSetDimensions(eps, p, PETSC_DEFAULT, PETSC_DEFAULT);
    EPSSetWhichEigenpairs(eps, EPS_LARGEST_MAGNITUDE);
    EPSSetFromOptions(eps);

    EPSSolve(eps);
    PetscPrintf(PETSC_COMM_WORLD, "%fs\n", MPI_Wtime() - start_eps);

    // Put eigenvalues and eigenvectors into Mat structure
    PetscScalar* values;
    PetscInt* row_indices;
    PetscScalar eigval;
    Vec eigvec;
    VecCreate(PETSC_COMM_WORLD, &eigvec);
    VecSetSizes(eigvec, PETSC_DECIDE, sample_size);
    VecSetFromOptions(eigvec);
    VecAssemblyBegin(eigvec);
    VecAssemblyEnd(eigvec);
    Mat Phi_A, Pi, Pi_Inv;
    MatCreate(PETSC_COMM_WORLD, &Phi_A);
    MatSetSizes(Phi_A, PETSC_DECIDE, PETSC_DECIDE, sample_size, p);
    MatSetType(Phi_A, MATMPIDENSE);
    MatSetFromOptions(Phi_A);
    MatSetUp(Phi_A);
    MatCreate(PETSC_COMM_WORLD, &Pi);
    MatSetSizes(Pi, PETSC_DECIDE, PETSC_DECIDE, p, p);
    MatSetType(Pi, MATMPIAIJ);
    MatSetFromOptions(Pi);
    MatSetUp(Pi);
    MatCreate(PETSC_COMM_WORLD, &Pi_Inv);
    MatSetSizes(Pi_Inv, PETSC_DECIDE, PETSC_DECIDE, p, p);
    MatSetType(Pi_Inv, MATMPIAIJ);
    MatSetFromOptions(Pi_Inv);
    MatSetUp(Pi_Inv);

    PetscInt istart, iend;
    VecGetOwnershipRange(eigvec, &istart, &iend);
    values = (PetscScalar*) malloc(sizeof(PetscScalar) * (iend-istart));
    row_indices = (PetscInt*) malloc(sizeof(PetscInt) * (iend-istart));
    for (unsigned int i = 0; i < (iend-istart); ++i)
    {
        row_indices[i] = i+istart;
    }
    for (unsigned int i = 0; i < p; ++i)
    {
        EPSGetEigenpair(eps, i, &eigval, NULL, eigvec, NULL);
        VecGetValues(eigvec, iend-istart, row_indices, values);
        MatSetValues(Phi_A, iend-istart, row_indices, 1, (int*)&i, values, INSERT_VALUES);
        MatSetValues(Pi, 1, (int*)&i, 1, (int*)&i, &eigval, INSERT_VALUES); // Diagonal
        eigval = 1./eigval;
        MatSetValues(Pi_Inv, 1, (int*)&i, 1, (int*)&i, &eigval, INSERT_VALUES); // Diagonal 1/eigval
    }
    VecDestroy(&eigvec);
    free(row_indices);
    free(values);
    EPSDestroy(&eps);

    MatAssemblyBegin(Phi_A, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(Pi, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(Pi_Inv, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Phi_A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Pi, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Pi_Inv, MAT_FINAL_ASSEMBLY);

    // Nyström
    Mat phi = Nystroem(K_B, Phi_A, Pi_Inv, width*height, sample_size, p);
    MatDestroy(&Phi_A);
    MatDestroy(&Pi_Inv);

    // Affinity = phi*Pi*phiT
    // TODO permutation doesn't quite work yet
    //Mat phi_perm = Permutation(phi, sample_indices, sample_size);
    //ComputeAndSaveAffinityMatrixOfPixel(phi_perm, Pi, width, height, 0, 1);
    //MatDestroy(&phi_perm);
    ComputeAndSaveAffinityMatrixOfPixel(phi, Pi, width, height, 0, 1);

    // End
    PetscPrintf(PETSC_COMM_WORLD, "Total computation time: %fs\n", MPI_Wtime() - start_time);

    // Clean up
    MatDestroy(&phi);
    MatDestroy(&Pi);
    MatDestroy(&K_A);
    MatDestroy(&K_B);
    free(sample_indices);

    for (unsigned int i = 0; i < height; ++i)
    {
        free(img_bytes[i]);
    }
    free(img_bytes);

    SlepcFinalize();

    return 0;
}
