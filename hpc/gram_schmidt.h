#include <petscvec.h>
#include <petscmat.h>

void OrthonormaliseVecs(Vec* X, const unsigned int n, const unsigned int p, PetscScalar* norms);
void NormaliseVecs(Vec* X, const unsigned int p, PetscScalar* norms);
Mat OrthonormaliseMat(Mat X);
Mat OrthogonaliseMat(Mat X);
