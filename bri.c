#include "bri.h"
#include "utils.h"
#include <assert.h>
#include <mkl.h>
#include <stdio.h>

struct bri_param {
  blockload_fn_t block_load;
  unsigned int bsize;
  unsigned int *Brows, *Bcols;
};

void bri_SA(double *Blo, unsigned int numb, unsigned int brow,
            unsigned int bcol, const struct bri_param *p);
void bri_SB(double *Blo, unsigned int numb, unsigned int brow,
            unsigned int bcol, const struct bri_param *p);
void bri_SC(double *Blo, unsigned int numb, unsigned int brow,
            unsigned int bcol, const struct bri_param *p);
void bri_SD(double *Blo, unsigned int numb, unsigned int brow,
            unsigned int bcol, const struct bri_param *p);

void bri_allocBlock(double **Blo, unsigned int bsize) {
  *Blo = (double *)malloc(sizeof(double) * bsize * bsize);
}

void bri_allocVec(unsigned int **vec, unsigned int size) {
  *vec = (unsigned int *)malloc(sizeof(unsigned int) * size);
}

lapack_int bri_matInv(double *A, unsigned n) {
  lapack_int info, ipiv[n + 1];

  info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, A, n, ipiv);
  if (info != 0) {
    perror("LU factorization failed");
    exit(-1);
  }

  info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, A, n, ipiv);
  if (info != 0) {
    perror("Inversion failed");
    exit(-1);
  } else
    return info;
}

void bri_initPvec(unsigned int *vec, unsigned int count) {
  for (unsigned int i = 0; i < count; ++i)
    vec[i] = i;
}

// Compute SA() - SB() * inv(SD()) * SC()
void bri_SA(double *P, unsigned int k, unsigned int i, unsigned int j,
            const struct bri_param *par) {
  double *Q = NULL, *R = NULL;
  unsigned int b = par->bsize;

  if (k == 1) {
    par->block_load(P, par->Brows[i], par->Bcols[j]);
    return;
  }
  bri_SB(P, k - 1, i, j + 1, par);
  bri_allocBlock(&Q, b);
  bri_SD(Q, k - 1, i + 1, j + 1, par);
  bri_matInv(Q, b);
  bri_allocBlock(&R, b);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, b, b, b, 1.0, P, b, Q,
              b, 0.0, R, b);
  free(Q);
  bri_SC(P, k - 1, i + 1, j, par);
  bri_allocBlock(&Q, b);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, b, b, b, 1.0, R, b, P,
              b, 0.0, Q, b);
  free(R);
  bri_SA(P, k - 1, i, j, par);
  for (unsigned int i = 0; i < b; i++)
    for (unsigned int j = 0; j < b; j++)
      P[i + b * j] -= Q[i + b * j];
  free(Q);
}

// Compute SB() - SA() * inv(SC()) * SD()
void bri_SB(double *P, unsigned int k, unsigned int i, unsigned int j,
            const struct bri_param *par) {
  double *Q = NULL, *R = NULL;
  unsigned int b = par->bsize;

  if (k == 1) {
    par->block_load(P, par->Brows[i], par->Bcols[j]);
    return;
  }
  bri_SA(P, k - 1, i, j, par);
  bri_allocBlock(&Q, b);
  bri_SC(Q, k - 1, i + 1, j, par);
  bri_matInv(Q, b);
  bri_allocBlock(&R, b);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, b, b, b, 1.0, P, b, Q,
              b, 0.0, R, b);
  free(Q);
  bri_SD(P, k - 1, i + 1, j + 1, par);
  bri_allocBlock(&Q, b);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, b, b, b, 1.0, R, b, P,
              b, 0.0, Q, b);
  free(R);
  bri_SB(P, k - 1, i, j + 1, par);
  for (unsigned int i = 0; i < b; i++)
    for (unsigned int j = 0; j < b; j++)
      P[i + b * j] -= Q[i + b * j];
  free(Q);
}

// Compute SC() - SD() * inv(SB()) * SA()
void bri_SC(double *P, unsigned int k, unsigned int i, unsigned int j,
            const struct bri_param *par) {
  double *Q = NULL, *R = NULL;
  unsigned int b = par->bsize;

  if (k == 1) {
    par->block_load(P, par->Brows[i], par->Bcols[j]);
    return;
  }
  bri_SD(P, k - 1, i + 1, j + 1, par);
  bri_allocBlock(&Q, b);
  bri_SB(Q, k - 1, i, j + 1, par);
  bri_matInv(Q, b);
  bri_allocBlock(&R, b);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, b, b, b, 1.0, P, b, Q,
              b, 0.0, R, b);
  free(Q);
  bri_SA(P, k - 1, i, j, par);
  bri_allocBlock(&Q, b);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, b, b, b, 1.0, R, b, P,
              b, 0.0, Q, b);
  free(R);
  bri_SC(P, k - 1, i + 1, j, par);
  for (unsigned int i = 0; i < b; i++)
    for (unsigned int j = 0; j < b; j++)
      P[i + b * j] -= Q[i + b * j];
  free(Q);
}

// Compute SD() - SC() * inv(SA()) * SB()
void bri_SD(double *P, unsigned int k, unsigned int i, unsigned int j,
            const struct bri_param *par) {
  double *Q = NULL, *R = NULL;
  unsigned int b = par->bsize;

  if (k == 1) {
    par->block_load(P, par->Brows[i], par->Bcols[j]);
    return;
  }
  bri_SC(P, k - 1, i + 1, j, par);
  bri_allocBlock(&Q, b);
  bri_SA(Q, k - 1, i, j, par);
  bri_matInv(Q, b);
  bri_allocBlock(&R, b);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, b, b, b, 1.0, P, b, Q,
              b, 0.0, R, b);
  free(Q);
  bri_SB(P, k - 1, i, j + 1, par);
  bri_allocBlock(&Q, b);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, b, b, b, 1.0, R, b, P,
              b, 0.0, Q, b);
  free(R);
  bri_SD(P, k - 1, i + 1, j + 1, par);
  for (unsigned int i = 0; i < b; i++)
    for (unsigned int j = 0; j < b; j++)
      P[i + b * j] -= Q[i + b * j];
  free(Q);
}

/**
 * @brief
 *
 * BRI function
 *
 * @param[out] P Pointer to inverted bsize x bsize block
 * @param[in] order - order of the block matrix
 * @param[in] bsize - size of each block
 * @param[in] numb - number of blocks per row/column
 * @param[in] brow, bcol - block row and column to be inverted
 * @param[in] blockload - function to load a block of the block matrix
 */
void bri(double *P, unsigned int order, unsigned int bsize, unsigned int numb,
         unsigned int brow, unsigned int bcol, blockload_fn_t blockload) {
  unsigned int aux;
  struct bri_param par;

  if (bsize * numb != order) {
    fprintf(stderr, "Error: bsize * numb !=  order\n");
    exit(-1);
  }

  par.block_load = blockload;
  par.bsize = bsize;
  bri_allocVec(&par.Brows, numb);
  bri_allocVec(&par.Bcols, numb);
  bri_initPvec(par.Bcols, numb);
  bri_initPvec(par.Brows, numb);

  aux = par.Brows[bcol];
  par.Brows[bcol] = par.Brows[0];
  par.Brows[0] = aux;
  aux = par.Bcols[brow];
  par.Bcols[brow] = par.Bcols[0];
  par.Bcols[0] = aux;

  bri_SA(P, numb, 0, 0, &par);
  bri_matInv(P, bsize);

  free(par.Brows);
  free(par.Bcols);
}
