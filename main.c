#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bri.h"
#include "utils.h"

static unsigned int order, numb, bsize;
static double *A = NULL;

void blockload(double *Blo, unsigned int brow, unsigned int bcol) {
  getBlock((double *)A, order, Blo, bsize, brow, bcol);
}

int main(int argc, char **argv) {
  unsigned int limit = 3, brow, bcol;

  order = 1000;
  numb = 5;
  brow = 1;
  bcol = 2;
  bsize = order / numb;

  printf("=========================================\n");
  printf("Inversion of a block of a matrix\n");
  printf("=========================================\n\n");
  printf("Matrix order.............: %d x %d\n", order, order);
  printf("Block order..............: %d x %d\n", bsize, bsize);
  printf("Number of blocks per row.: %d\n", numb);
  printf("Block row................: %d\n", brow);
  printf("Block column.............: %d\n", bcol);

  printf(
      "\nPrinting only the %d x %d top-left submatrix of the inverted block\n",
      limit, limit);

  A = (double *)malloc(sizeof(double) * order * order);
  randMat(A, order);

  double *iA = NULL;
  allocMat(&iA, order);
  memcpy(iA, A, order * order * sizeof(double));
  if (matInv(iA, order) < 0) {
    perror("Error inverting matrix");
  }
  printf("\nInverted by LU\n");
  printBlock(iA, order, bsize, brow, bcol, limit);
  free(iA);

  double *iAb = NULL;
  allocMat(&iAb, bsize);
  bri(iAb, order, bsize, numb, brow, bcol, blockload);
  printf("\nInverted by BRI\n");
  printBlock(iAb, bsize, bsize, 0, 0, limit);
  free(iAb);

  free(A);
  return 0;
}