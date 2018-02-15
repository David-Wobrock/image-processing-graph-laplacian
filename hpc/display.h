#include <petscmat.h>
#include <petscvec.h>

#include <png.h>

void ComputeAndSaveResult(const png_bytep* const img_bytes, Mat phi, Mat Pi, const unsigned int width, const unsigned int height);

void WriteVec(Vec v, const char* const filename);
void WriteDiagMat(Mat x, const char* const filename);
void WriteMatCol(Mat x, const unsigned int col_num, const char* const filename);
void WritePngMatCol(Mat x, const unsigned int col_num, const unsigned int width, const unsigned int height, const char* const filename);

png_bytep* ComputeResultFromLaplacian(const png_bytep* const img_bytes, Mat phi, Mat Pi, const unsigned int width, const unsigned int height);
