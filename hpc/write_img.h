#include <petscmat.h>
#include <png.h>

png_bytep* VecMat2pngbytes(Mat diag_mat, const unsigned int width, const unsigned int height, const int scale);
int write_png(const char* const filename, png_bytep* img_bytes, const unsigned int width, const unsigned int height);
