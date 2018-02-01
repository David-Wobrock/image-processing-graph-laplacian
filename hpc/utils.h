#include <petscvec.h>
#include <petscmat.h>
#include <png.h>

extern const PetscInt ZERO;

unsigned int num2x(const unsigned int num, const unsigned int num_col);
unsigned int num2y(const unsigned int num, const unsigned int num_col);
unsigned int xy2num(const unsigned int x, const unsigned y, const unsigned int num_col);
void Vecs2Mat(Vec* vecs, Mat* m, const unsigned int ncols);
Vec* Mat2Vecs(Mat m);
Mat OneColMat2Diag(Mat x);
Mat pngbytes2OneColMat(const png_bytep* const img_bytes, const unsigned int width, const unsigned int height);
png_bytep* OneColMat2pngbytes(Mat x, const unsigned int width, const unsigned int height);
png_bytep* OneRowMat2pngbytes(Mat vec_mat, const unsigned int width, const unsigned int height, const int scale);
Vec DiagMat2Vec(Mat x);

Mat Permutation(Mat m, const unsigned int* const sample_indices, const unsigned int num_sample_indices);
Mat GetFirstCols(Mat x, const unsigned int n);
Mat GetLastCols(Mat x, const unsigned int n);
Mat GetFirstRows(Mat x, const unsigned int n);

Mat SetNegativesToZero(Mat x);

Vec MatRowSum(Mat A);
PetscScalar VecMean(Vec x);
Mat MatCreateIdentity(const unsigned int n, const MatType format);

void CopyVecs(Vec* in, Vec* out, const unsigned int n);
