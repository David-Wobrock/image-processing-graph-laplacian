#include <petscmat.h>

Mat ComputeK(Mat phi, Mat Pi, const unsigned int sample_size);
void ComputeWAWB_RenormalisedLaplacian(Mat phi, Mat Pi, Mat* W_A, Mat* W_B, const unsigned int sample_size);
