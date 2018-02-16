#include "display.h"

#include <mpi.h>
#include <petscviewer.h>

#include "utils.h"
#include "write_img.h"

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

void WriteMatCol(Mat x, const unsigned int col_num, const char* const filename)
{
    PetscInt n;
    MatGetSize(x, &n, NULL);

    Vec v;
    VecCreate(PETSC_COMM_WORLD, &v);
    VecSetSizes(v, PETSC_DECIDE, n);
    VecSetFromOptions(v);

    MatGetColumnVector(x, v, col_num);

    WriteVec(v, filename);

    VecDestroy(&v);
}

void WritePngMatCol(Mat x, const unsigned int col_num, const unsigned int width, const unsigned int height, const char* const filename)
{
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    // Get Mat with one col
    Mat tmp = GetFirstCols(x, col_num);
    Mat eigvec = GetLastCols(tmp, 1);
    MatDestroy(&tmp);

    // Get img
    png_bytep* img = OneColMat2pngbytes(eigvec, width, height);
    MatDestroy(&eigvec);

    // Process 0 writes and deletes
    if (rank == 0)
    {
        write_png(filename, img, width, height);
        for (unsigned int i = 0; i < height; ++i)
        {
            free(img[i]);
        }
        free(img);
    }
}

png_bytep* ComputeResultFromEntireLaplacian(const png_bytep* const img_bytes, Mat Lapl, const unsigned int width, const unsigned int height)
{
    Mat z = pngbytes2OneColMat(img_bytes, width, height);

    Mat Lapl_y;

    MatMatMult(Lapl, z, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Lapl_y);

    // z = y - Ly
    MatAXPY(z, -1.0, Lapl_y, SAME_NONZERO_PATTERN);
    MatDestroy(&Lapl_y);

    Mat z_tmp = AboveXSetY(z, 255, 255);
    MatDestroy(&z);
    z = SetNegativesToZero(z_tmp);
    MatDestroy(&z_tmp);

    png_bytep* output = OneColMat2pngbytes(z, width, height);
    MatDestroy(&z);
    return output;
}
