#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "spqlios/arithmetic/vec_znx_arithmetic.h"
#include "spqlios/q120/q120_common.h"

static int cpu_supports_avx2(void) {
#if defined(__x86_64__) || defined(_M_X64)
  return __builtin_cpu_supports("avx2");
#else
  return 0;
#endif
}

static void print_i128(__int128_t v) {
  if (v == 0) {
    printf("0");
    return;
  }
  if (v < 0) {
    putchar('-');
    v = -v;
  }
  char buf[64];
  int i = 0;
  while (v > 0) {
    buf[i++] = (char)('0' + (v % 10));
    v /= 10;
  }
  while (i-- > 0) {
    putchar(buf[i]);
  }
}

static void negacyclic_naive(const int64_t* a, const int64_t* b, __int128_t* out, uint64_t n) {
  for (uint64_t i = 0; i < n; i++) {
    __int128_t acc = 0;
    for (uint64_t j = 0; j < n; j++) {
      uint64_t k = (i + n - j) & (n - 1);
      __int128_t term = (__int128_t)a[j] * b[k];
      if (j > i) {
        acc -= term;
      } else {
        acc += term;
      }
    }
    out[i] = acc;
  }
}

static void pointwise_mul_q120(uint64_t nn, uint64_t* out, const uint64_t* a, const uint64_t* b) {
  const uint64_t mods[4] = {Q1, Q2, Q3, Q4};
  for (uint64_t i = 0; i < nn; i++) {
    for (uint64_t k = 0; k < 4; k++) {
      uint64_t av = a[i * 4 + k] % mods[k];
      uint64_t bv = b[i * 4 + k] % mods[k];
      __uint128_t prod = (__uint128_t)av * (__uint128_t)bv;
      out[i * 4 + k] = (uint64_t)(prod % mods[k]);
    }
  }
}

int main(void) {
  const uint64_t N = 16;
  const uint64_t size = 5;    // multiple polynomials
  const uint64_t stride = N;  // coefficients per polynomial (contiguous)

  if (!cpu_supports_avx2()) {
    fprintf(stderr, "NTT120 backend requires AVX2 on this build.\n");
    return 2;
  }

  MODULE* module = new_module_info(N, NTT120);
  if (!module) {
    fprintf(stderr, "failed to create module\n");
    return 1;
  }

  int64_t* a = (int64_t*)calloc(size * stride, sizeof(int64_t));
  int64_t* b = (int64_t*)calloc(size * stride, sizeof(int64_t));
  if (!a || !b) {
    fprintf(stderr, "allocation failed\n");
    return 1;
  }

  for (uint64_t p = 0; p < size; p++) {
    for (uint64_t i = 0; i < N; i++) {
      a[p * stride + i] = (int64_t)(i + 1 + p);
      b[p * stride + i] = (int64_t)(N - i + p);
    }
  }

  VEC_ZNX_DFT* a_dft = new_vec_znx_dft(module, size);
  VEC_ZNX_DFT* b_dft = new_vec_znx_dft(module, size);
  VEC_ZNX_DFT* c_dft = new_vec_znx_dft(module, size);
  VEC_ZNX_BIG* c_big = new_vec_znx_big(module, size);
  if (!a_dft || !b_dft || !c_dft || !c_big) {
    fprintf(stderr, "allocation failed\n");
    return 1;
  }

  vec_znx_dft(module, a_dft, size, a, size, stride);
  vec_znx_dft(module, b_dft, size, b, size, stride);

  uint64_t* a_u64 = (uint64_t*)a_dft;
  uint64_t* b_u64 = (uint64_t*)b_dft;
  uint64_t* c_u64 = (uint64_t*)c_dft;
  for (uint64_t p = 0; p < size; p++) {
    pointwise_mul_q120(N, c_u64 + p * N * 4, a_u64 + p * N * 4, b_u64 + p * N * 4);
  }

  uint64_t tmp_bytes = vec_znx_idft_tmp_bytes(module);
  uint8_t* tmp = (uint8_t*)malloc(tmp_bytes);
  if (!tmp) {
    fprintf(stderr, "tmp allocation failed\n");
    return 1;
  }
  vec_znx_idft(module, c_big, size, c_dft, size, tmp);

  printf("batch size=%" PRIu64 ", N=%" PRIu64 "\n", size, N);
  for (uint64_t p = 0; p < size; p++) {
    __int128_t naive[N];
    negacyclic_naive(a + p * stride, b + p * stride, naive, N);

    printf("poly %" PRIu64 " result (NTT):\n", p);
    __int128_t* out = (__int128_t*)c_big + p * N;
    for (uint64_t i = 0; i < N; i++) {
      print_i128(out[i]);
      putchar(i + 1 == N ? '\n' : ' ');
    }

    printf("poly %" PRIu64 " result (naive):\n", p);
    for (uint64_t i = 0; i < N; i++) {
      print_i128(naive[i]);
      putchar(i + 1 == N ? '\n' : ' ');
    }
  }

  free(tmp);
  free(a);
  free(b);
  delete_vec_znx_dft(a_dft);
  delete_vec_znx_dft(b_dft);
  delete_vec_znx_dft(c_dft);
  delete_vec_znx_big(c_big);
  delete_module_info(module);
  return 0;
}
