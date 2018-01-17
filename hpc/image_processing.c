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
#include "utils.h"
#include "affinity.h"
#include "eigendecomposition.h"
#include "nystroem.h"
#include "filter.h"
#include "gram_schmidt.h"
#include "display.h"

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
    //unsigned int sample_size = 500;
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
    //const unsigned int p = sample_size; // num eigenpairs
    const unsigned int p = 100; // num eigenpairs
    PetscPrintf(PETSC_COMM_WORLD, "Computing %d largest eigenvalues of affinity matrix... ", p);
    Mat phi_A, Pi, Pi_Inv;
    Eigendecomposition(K_A, p, &phi_A, &Pi, &Pi_Inv, NULL); // A = phi*Pi*phi_T
    PetscPrintf(PETSC_COMM_WORLD, "%fs\n", MPI_Wtime() - start_eps);
    MatDestroy(&K_A);

    // Nyström
    double start_nystroem = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD, "Computing Nyström approximation... ");
    Mat phi = Nystroem(K_B, phi_A, Pi_Inv, width*height, sample_size, p);
    PetscPrintf(PETSC_COMM_WORLD, "%fs\n", MPI_Wtime() - start_nystroem);
    MatDestroy(&phi_A);
    MatDestroy(&Pi_Inv);
    MatDestroy(&K_B);

    // Display affinity = phi*Pi*phiT
    const unsigned int display_x = 55;
    const unsigned int display_y = 425;
    PetscPrintf(PETSC_COMM_WORLD, "Displaying affinity matrix at %dx%d... ", display_x, display_y);
    Mat phi_perm = Permutation(phi, sample_indices, sample_size);
    ComputeAndSaveAffinityMatrixOfPixel(phi_perm, Pi, width, height, display_x, display_y);
    MatDestroy(&phi_perm);
    PetscPrintf(PETSC_COMM_WORLD, "done\n");

    // Compute W_A and W_B with re-normalised Laplacian
    double start_filter = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD, "Computing W_A and W_B (re-normalised Laplacian)... ");
    Mat W_A, W_B;
    ComputeWAWB_RenormalisedLaplacian(phi, Pi, &W_A, &W_B);
    Mat W_A_tmp = SetNegativesToZero(W_A);
    MatDestroy(&W_A);
    W_A = W_A_tmp;
    PetscPrintf(PETSC_COMM_WORLD, "%fs\n", MPI_Wtime() - start_filter);
    MatDestroy(&phi);
    MatDestroy(&Pi);

    // Eigendecomposition and Nystroem
    start_eps = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD, "Computing %d largest eigenvalues of filter matrix... ", p);
    Eigendecomposition(W_A, p, &phi_A, &Pi, &Pi_Inv, NULL); // A = phi*Pi*phi_T
    MatView(Pi, PETSC_VIEWER_STDOUT_WORLD);
    PetscPrintf(PETSC_COMM_WORLD, "%fs\n", MPI_Wtime() - start_eps);
    MatDestroy(&W_A);

    // Nyström
    start_nystroem = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD, "Computing Nyström approximation... ");
    phi = Nystroem(W_B, phi_A, Pi_Inv, width*height, sample_size, p);
    PetscPrintf(PETSC_COMM_WORLD, "%fs\n", MPI_Wtime() - start_nystroem);
    MatDestroy(&phi_A);
    MatDestroy(&Pi_Inv);
    MatDestroy(&K_B);

    // TODO necessary?
    //Mat phi_orth = OrthogonaliseMat(phi);
    //MatDestroy(&phi);
    //phi = phi_orth;

    // Can display
    double start_output = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD, "Permuting and computing output image... ");
    phi_perm = Permutation(phi, sample_indices, sample_size);
    ComputeAndSaveResult(img_bytes, phi_perm, Pi, width, height);
    PetscPrintf(PETSC_COMM_WORLD, "%fs\n", MPI_Wtime() - start_output);
    MatDestroy(&phi_perm);
    MatDestroy(&phi);
    MatDestroy(&Pi);

    // End
    PetscPrintf(PETSC_COMM_WORLD, "Total computation time: %fs\n", MPI_Wtime() - start_time);

    // Clean up
    free(sample_indices);

    for (unsigned int i = 0; i < height; ++i)
    {
        free(img_bytes[i]);
    }
    free(img_bytes);

    SlepcFinalize();

    return 0;
}
