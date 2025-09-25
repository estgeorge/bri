#ifndef HEADER_BRI
#define HEADER_BRI

typedef void (*blockload_fn_t)(double *, unsigned int, unsigned int);

void bri(double *P, unsigned int order, unsigned int bsize, unsigned int numb,
         unsigned int brow, unsigned int bcol, blockload_fn_t blockload);

void bri_allocBlock(double **Blo, unsigned int bsize);

#endif
