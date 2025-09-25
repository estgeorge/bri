#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void randMat(double *Mat, unsigned int order) {
  // srand(time(NULL));
  srand(0);
  for (unsigned int i = 0; i < order * order; i++) {
    Mat[i] =
        ((double)(rand() % 1000) / 1000) * ((double)(rand() % 1000) / 1000);
  }
}

void allocMat(double **Mat, unsigned int order) {
  *Mat = (double *)malloc(sizeof(double) * order * order);
}

lapack_int matInv(double *A, unsigned n) {
  lapack_int ipiv[n + 1];
  lapack_int ret;

  ret = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, A, n, ipiv);

  if (ret != 0) {
    perror("LU factorization failed");
    exit(-1);
  }

  ret = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, A, n, ipiv);
  if (ret != 0) {
    perror("Inversion failed");
    exit(-1);
  } else
    return ret;
}

void printBlock(double *Mat, unsigned int order, unsigned int bsize,
                unsigned int brow, unsigned int bcol, unsigned int limit) {
  putchar('\n');
  for (unsigned int i = 0; i < limit; ++i) {
    for (unsigned int j = 0; j < limit; ++j)
      printf("%+6.3f ", Mat[(brow * bsize + i) * order + bcol * bsize + j]);
    putchar('\n');
  }
}

void getBlock(double *Mat, unsigned int order, double *Blo, unsigned int bsize,
              unsigned int brow, unsigned int bcol) {
  for (unsigned int i = 0; i < bsize; ++i)
    for (unsigned int j = 0; j < bsize; ++j)
      Blo[i * bsize + j] = Mat[(brow * bsize + i) * order + bcol * bsize + j];
}
