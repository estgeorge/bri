# BRI algorithm in C

Implementation of the block recursive inversion algorithm from article:
```
@article{cosme2018memory,
  title={Memory-usage advantageous block recursive matrix inverse},
  author={Cosme, Iria CS and Fernandes, Isaac F and de Carvalho, Jo{\~a}o L and Xavier-de-Souza, Samuel},
  journal={Applied Mathematics and Computation},
  volume={328},
  pages={125--136},
  year={2018},
  publisher={Elsevier}
}
```

In this implementation I omitted the exchange of block rows and/or block columns described in step 2 of section 3.1. I found it unnecessary for my work.


## Requirements

- Intel C++ Compiler;
- Intel Math Kernel Library (MKL).

## Description

Given a `order` x `order` matrix `M` with `numb` rows/columns of blocks. Then function `bri()` returns a `bsize` x `bsize` block `Blo` which is the inverse of the (`nrow`, `ncol`) block of the matrix `M`.

All functions in this program use 1D array with row-major order to simulate a 2D matrix.

Example:

```
Blo = (double *)malloc(sizeof(double) * numb * numb);

bri(Blo, order, bsize, numb, brow, bcol, blockload);

// Process the block Blo.

free(Blo);
```
where `blockload()` is a user-defined function that loads an arbitrary block from the matrix `M` given the block indices `brow` and `bcol`. Example:
```
void blockload(double *Blo, unsigned int brow, unsigned int bcol) {

  // Write here the logic of the code that loads the (brow, bcol) block 
  // of the matrix M into the array Blo.
  
}  
```
  

## Example

Compile:
```
icx main.c bri.c utils.c -o main -qmkl=parallel 
```
Execute:

```
export OMP_NUM_THREADS=N; ./main
```
Replace `N` with an appropriate number of computer cores.
