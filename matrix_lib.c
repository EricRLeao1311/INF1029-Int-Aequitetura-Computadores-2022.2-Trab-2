/*INF1029 - INT ARQUITETURA COMPUTADORES - 2022.2 - 3WA
Trabalho 2 - Módulo avançado (AVX/FMA) para operações com matrizes
Nome: Eric Leão     Matrícula: 2110694
Nome: Pedro Machado Peçanha    Matrícula: 2110535*/

#include <assert.h>
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>

struct matrix {
  unsigned long int height;
  unsigned long int width;
  float *rows;
};

int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
  if (matrix == NULL)
    return 0;
  int matrixSize = (matrix->height) * (matrix->width);
  __m256 vec;
  __m256 vec1 = _mm256_set1_ps(scalar_value);
  for (int c = 0; c < matrixSize; c += 8) {
    vec = _mm256_load_ps(matrix->rows + c);
    vec = _mm256_mul_ps(vec, vec1);
    _mm256_store_ps(matrix->rows + c, vec);
  }
  return 1;
}

int matrix_matrix_mult(struct matrix *matrixA, struct matrix *matrixB,
                       struct matrix *matrixC) {
  if (matrixA == NULL)
    return 0; // matrx A diferente de NULL
  if (matrixB == NULL)
    return 0; // matrx B diferente de NULL
  if (matrixC == NULL)
    return 0; // matrx C diferente de NULL
  if (matrixA->width != matrixB->height)
    return 0; // matrx A precisa ter largura igual à altura da matrix B
  if (matrixA->height != matrixC->height || matrixB->width != matrixC->width)
    return 0; // matriz C tem que ter altura e largura compativeis com a amatriz
              // A e B
  if((matrixA->height%8)!=0 || (matrixB->height%8)!=0 || (matrixB->width%8)!=0)
    return 0;
  __m256 vec0;
  __m256 vec1;
  __m256 vec2;
  __m256 vecZero = _mm256_setzero_ps();
  for (int i = 0; i < matrixA->height; i++) {
    for (int j = 0; j < matrixA->width; j++) {

      vec1 = _mm256_set1_ps(*(matrixA->rows + i * matrixA->width + j));
      for (int k = 0; k < matrixB->width; k += 8) {
        if (j == 0) { // testa se esta passando pela primeira vez no elemento
                      // damatrix
          _mm256_store_ps(matrixC->rows + i * matrixC->width + k, vecZero);
          }
        vec0 = _mm256_load_ps(matrixC->rows + i * matrixC->width + k);
        vec2 = _mm256_load_ps(matrixB->rows + j * matrixB->width + k);
        vec0 = _mm256_fmadd_ps(vec1, vec2, vec0);
        _mm256_store_ps(matrixC->rows + i * matrixC->width + k, vec0);
      }
    }
  }
  return 1;
}
