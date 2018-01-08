#include <petscvec.h>
#include <petscmat.h>

void OrthonormaliseVecs(Vec* X, const unsigned int n, const unsigned int p);
Mat OrthonormaliseMat(Mat X);
