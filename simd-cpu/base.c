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

static void add(const float *a, const float *b, float *c, size_t n)
{
  for (size_t i = 0; i < n; ++i) { c[i] = a[i] + b[i]; }
}

int main()
{
  const size_t n = 8000000; // 8M elements
  float *a = (float*)malloc(n * sizeof(float));
  float *b = (float*)malloc(n * sizeof(float));
  float *c = (float*)malloc(n * sizeof(float));
  if (!a || !b || !c) return 1;
  // INIT
  for (size_t i = 0; i < n; ++i) { a[i] = (float)i; b[i] = (float)(2*i); }

  for (int i = 0; i < 20; ++i)
  {
    // RUNTIME
    uint64_t st = get_time();
    // WORK
    add(a, b, c, n);
    uint64_t et = get_time();
    double dur = (double)(et - st)/1e9;
    // DEBUG
    printf("(BASE) Runtime : %.4fs\n", dur);
  }
  // CLEANUP
  free(a); free(b); free(c);
  return 0;
}
