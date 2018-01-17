#include <petscmat.h>

#include <png.h>

void ComputeAndSaveAffinityMatrixOfPixel(Mat phi, Mat Pi, const unsigned int width, const unsigned int height, const unsigned int pixel_x, const unsigned int pixel_y);

void ComputeAndSaveResult(const png_bytep* const img_bytes, Mat phi, Mat Pi, const unsigned int width, const unsigned int height);
