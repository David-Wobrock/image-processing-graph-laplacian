#include <petscvec.h>
#include <petscmat.h>

unsigned int num2x(const unsigned int num, const unsigned int num_col);
unsigned int num2y(const unsigned int num, const unsigned int num_col);
unsigned int xy2num(const unsigned int x, const unsigned y, const unsigned int num_col);
void Vecs2Mat(Vec* vecs, Mat* m, const unsigned int ncols);
Vec* Mat2Vecs(Mat m);

Mat Permutation(Mat m, const unsigned int* const sample_indices, const unsigned int num_sample_indices);
