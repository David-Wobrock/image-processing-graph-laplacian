#include "display.h"

#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <petscviewer.h>

#include "utils.h"
#include "write_img.h"

static void ComputeAndSaveAffinityMatrixOfPixelNum(Mat phi, Mat Pi, const unsigned int width, const unsigned int height, const unsigned int pixel_num, const char* const filename)
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
        for (unsigned int i = 0; i < p; ++i)
        {
            MatSetValues(phi_Vec, 1, &ZERO, 1, (int*)&i, values+i, INSERT_VALUES);
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

    Mat non_zero_affinities = SetNegativesToZero(affinity_img_on_vec);
    MatDestroy(&affinity_img_on_vec);
    affinity_img_on_vec = non_zero_affinities;

    // Rearrange vector into image (on one proc) and save
    png_bytep* img_bytes = OneRowMat2pngbytes(affinity_img_on_vec, width, height, 255);
    MatDestroy(&affinity_img_on_vec);
    if (rank == 0)
    {
        write_png(filename, img_bytes, width, height);
        for (unsigned int i = 0; i < height; ++i)
        {
            free(img_bytes[i]);
        }
        free(img_bytes);
    }
}

void ComputeAndSaveAffinityMatrixOfPixel(Mat phi, Mat Pi, const unsigned int width, const unsigned int height, const unsigned int pixel_x, const unsigned int pixel_y)
{
    char filename[100], x_name[20], y_name[20];
    sprintf(x_name, "%d", pixel_x);
    sprintf(y_name, "%d", pixel_y);
    strcpy(filename, "results/affinity_");
    strcat(filename, x_name);
    strcat(filename, "x");
    strcat(filename, y_name);
    strcat(filename, ".png");

    ComputeAndSaveAffinityMatrixOfPixelNum(phi, Pi, width, height, xy2num(pixel_x, pixel_y, width), filename);
}

void ComputeAndSaveResult(const png_bytep* const img_bytes, Mat phi, Mat Pi, const unsigned int width, const unsigned int height)
{
    Mat img_mat = pngbytes2OneColMat(img_bytes, width, height);

    Mat phi_T_y;
    MatTransposeMatMult(phi, img_mat, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &phi_T_y);
    MatDestroy(&img_mat);

    Mat Pi_phi_T_y;
    MatMatMult(Pi, phi_T_y, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Pi_phi_T_y);
    MatDestroy(&phi_T_y);

    Mat result;
    MatMatMult(phi, Pi_phi_T_y, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &result);
    MatDestroy(&Pi_phi_T_y);

    png_bytep* result_bytes = OneColMat2pngbytes(result, width, height);
    MatDestroy(&result);

    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank == 0)
    {
        write_png("results/result.png", result_bytes, width, height);
        for (unsigned int i = 0; i < height; ++i)
        {
            free(result_bytes[i]);
        }
        free(result_bytes);
    }
}

void WriteVec(Vec v, const char* const filename)
{
    PetscViewer viewer;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename, &viewer);

    VecView(v, viewer);

    PetscViewerDestroy(&viewer);
}

void WriteDiagMat(Mat x, const char* const filename)
{
    Vec v = DiagMat2Vec(x);
    WriteVec(v, filename);
    VecDestroy(&v);
}

png_bytep* ComputeResultFromLaplacian(const png_bytep* const img_bytes, Mat phi, Mat Pi, const unsigned int width, const unsigned int height)
{
    Mat z = pngbytes2OneColMat(img_bytes, width, height);

    Mat phi_T, left, right, Lapl_y;

    MatMatMult(phi, Pi, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &left);
    MatTranspose(phi, MAT_INITIAL_MATRIX, &phi_T);
    MatMatMult(phi_T, z, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &right);
    MatDestroy(&phi_T);
    MatMatMult(left, right, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Lapl_y);
    MatDestroy(&left);
    MatDestroy(&right);

    MatAXPY(z, -1.0, Lapl_y, SAME_NONZERO_PATTERN);
    MatDestroy(&Lapl_y);

    Mat z_tmp = AboveXSetY(z, 255, 255);
    MatDestroy(&z);
    z = z_tmp;

    png_bytep* output = OneColMat2pngbytes(z, width, height);
    MatDestroy(&z);
    return output;
}
