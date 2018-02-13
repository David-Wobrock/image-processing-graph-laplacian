#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>
#include <png.h>

#include <petscsys.h>
#include <petscmat.h>
#include <petscvec.h>

#include <slepcsys.h>

#include "read_img.h"
#include "utils.h"
#include "sampling.h"
#include "affinity.h"
#include "eigendecomposition.h"
#include "nystroem.h"
#include "gram_schmidt.h"
#include "display.h"
#include "laplacian.h"
#include "inverse_power_it.h"
#include "write_img.h"

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
    if (!found_filename)
    {
        PetscFPrintf(PETSC_COMM_WORLD, stderr, "No filename found (option -f)\n");
        exit(1);
    }
    return ierr;
}

static PetscInt GetNumberEigenvalues(const unsigned int sample_size)
{
    PetscInt num_eigvals;
    PetscBool found;

    PetscOptionsGetInt(NULL, NULL, "-num_eigvals", &num_eigvals, &found);
    if (!found || num_eigvals < 0 || num_eigvals >= sample_size)
    {
        num_eigvals = sample_size-1;
        PetscFPrintf(PETSC_COMM_WORLD, stderr, "No or invalid number of eigenvalues found (option -num_eigvals), so using %d\n", num_eigvals);
    }
    return num_eigvals;
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

int main(int argc, char** argv)
{
    char filename[PETSC_MAX_PATH_LEN];
    PetscMPIInt rank, size;

    InitProgram(argc, argv, &rank, &size);
    PetscPrintf(PETSC_COMM_WORLD, "Running with %d processes\n", size);
    double start_time = MPI_Wtime();
    GetFilePath(filename);

    int width, height;
    png_bytep* img_bytes;
    ReadAndBcastImage(rank, filename, &img_bytes, &width, &height);
    PetscPrintf(PETSC_COMM_WORLD, "Read image %s of size %dx%d => %d pixels\n", filename, width, height, width*height);

    // * Define parameters
    unsigned int p; // Sample size
    p = width*height*0.01; // 1%
    //p = width*height*0.005; // 0.5%
    PetscInt m = GetNumberEigenvalues(p); // Number of eigenvalues

    // Sampling (all compute the same locally)
    unsigned int* sample_indices; // Must be sorted ASC
    Sampling(width, height, &p, &sample_indices);
    PetscPrintf(PETSC_COMM_WORLD, "Sample size: %d\n", p);

    // Compute affinity matrix
    double start_affinity = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD, "Computing affinity matrices... ");
    Mat K_A, K_B;
    ComputeAffinityMatrices(&K_A, &K_B, img_bytes, width, height, p, sample_indices);
    PetscPrintf(PETSC_COMM_WORLD, "%fs\n", MPI_Wtime() - start_affinity);

    // Compute Laplacian
    double start_laplacian = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD, "Computing Laplacian matrices... ");
    Mat L_A, L_B;
    ComputeLaplacianMatrix(&L_A, &L_B, K_A, K_B);
    PetscPrintf(PETSC_COMM_WORLD, "%fs\n", MPI_Wtime() - start_laplacian);
    MatDestroy(&K_A);
    MatDestroy(&K_B);

    // Eigendecomposition of L_A
    double start_inv_it = MPI_Wtime();
    Mat eigvals, eigvecs_A;
    PetscPrintf(PETSC_COMM_WORLD, "Computing %d smallest eigenvalues... ", m);
    InversePowerIteration(L_A, m, &eigvecs_A, &eigvals);
    //EigendecompositionSmallest(L_A, m, &eigvecs_A, &eigvals, NULL);
    PetscPrintf(PETSC_COMM_WORLD, "%fs\n", MPI_Wtime() - start_inv_it);
    WriteDiagMat(eigvals, "results/eigenvalues_laplacian.txt");
    MatDestroy(&L_A);

    Mat eigvals_inv = InverseDiagMat(eigvals);

    // Nyström
    double start_nystroem = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD, "Computing Nyström approximation... ");
    Mat eigvecs = Nystroem(L_B, eigvecs_A, eigvals_inv, width*height, p, m);
    PetscPrintf(PETSC_COMM_WORLD, "%fs\n", MPI_Wtime() - start_nystroem);
    MatDestroy(&eigvecs_A);
    MatDestroy(&eigvals_inv);
    MatDestroy(&L_B);

    // Permutation of eigenvectors
    Mat eigvecs_perm = Permutation(eigvecs, sample_indices, p);
    MatDestroy(&eigvecs);
    eigvecs = eigvecs_perm;
    WriteMatCol(eigvecs, 0, "results/eigenvector_0_laplacian.txt");
    WriteMatCol(eigvecs, 1, "results/eigenvector_1_laplacian.txt");
    WriteMatCol(eigvecs, 2, "results/eigenvector_2_laplacian.txt");

    // Apply some function to the eigenvalues
    Mat f_eigvals = MatPow(eigvals, 2);
    MatDestroy(&eigvals);

    // Compute output image z = y - (phi*Pi*phi.T*y)
    double start_result = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD, "Computing output image... ");
    png_bytep* output_img = ComputeResultFromLaplacian(img_bytes, eigvecs, f_eigvals, width, height);
    PetscPrintf(PETSC_COMM_WORLD, "%fs\n", MPI_Wtime() - start_result);
    MatDestroy(&eigvecs);
    MatDestroy(&f_eigvals);

    // Write image
    if (rank == 0)
    {
        write_png("results/output.png", output_img, width, height);
    }

    // End
    PetscPrintf(PETSC_COMM_WORLD, "Total computation time: %fs\n", MPI_Wtime() - start_time);

    // Clean up
    free(sample_indices);
    for (unsigned int i = 0; i < height; ++i)
    {
        free(img_bytes[i]);
    }
    free(img_bytes);

    if (rank == 0)
    {
        for (unsigned int i = 0; i < height; ++i)
        {
            free(output_img[i]);
        }
        free(output_img);
    }

    SlepcFinalize();

    return 0;
}
