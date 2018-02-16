#include <petscmat.h>

#include <png.h>

void ComputeAffinityMatrices(Mat* K_A, Mat* K_B, const png_bytep* const img_bytes, const int width, const int height, const unsigned int sample_size, const unsigned int* sample_indices);

void ComputeEntireAffinityMatrix(Mat* K, const png_bytep* const img_bytes, const int width, const int height);
