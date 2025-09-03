#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

static uint64_t get_time()
{
  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  return (uint64_t)start.tv_sec*1e9 + (uint64_t)start.tv_nsec;
}

static void add_avx2(const float *a, const float *b, float *c, size_t n)
{
  size_t i = 0;
  const size_t step = 8; // 8 floats per 256-bit AVX register

  for (; i + step <= n; i += step)
  {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 vc = _mm256_add_ps(va, vb);
    _mm256_storeu_ps(c + i, vc);
  }
}

int main(void)
{
  const size_t n = 8000000; // 8M elements
  float *a = (float*)aligned_alloc(32, n * sizeof(float));
  float *b = (float*)aligned_alloc(32, n * sizeof(float));
  float *c = (float*)aligned_alloc(32, n * sizeof(float));
  if (!a || !b || !c) return 1;
  // INIT
  for (size_t i = 0; i < n; ++i) { a[i] = (float)i; b[i] = (float)(2*i); }
  
  for (int i = 0; i < 20; ++i)
  {
    // RUNTIME
    uint64_t st = get_time();
    // WORK
    add_avx2(a, b, c, n);
    uint64_t et = get_time();
    double dur = (double)(et - st)/1e9;
    // DEBUG
    printf("(SIMD) Runtime : %.4fs\n", dur);
  }
  // CLEANUP
  free(a); free(b); free(c);
  return 0;
}
