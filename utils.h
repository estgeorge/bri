#ifndef _UTILS_H_
#define _UTILS_H_
#include <mkl.h>

void randMat(double *Mat, unsigned int order);
void allocMat(double **Mat, unsigned int order);
void printMat(double *Mat, unsigned int order);
lapack_int matInv(double *A, unsigned n);
void getBlock(double *Mat, unsigned int order, double *Blo, unsigned int bsize,
              unsigned int brow, unsigned int bcol);
void printBlock(double *Mat, unsigned int order, unsigned int bsize,
                unsigned int brow, unsigned int bcol, unsigned int limit);

#endif
