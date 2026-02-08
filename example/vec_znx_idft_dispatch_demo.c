#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "spqlios/arithmetic/vec_znx_arithmetic.h"

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

int main(void) {
  const uint64_t n = 1024;
  const uint64_t batch = 1;

  MODULE* module = new_module_info(n, NTT120);
  if (!module) {
    fprintf(stderr, "new_module_info failed\n");
    return 1;
  }

  int64_t* input = (int64_t*)calloc(batch * n, sizeof(int64_t));
  if (!input) {
    fprintf(stderr, "input allocation failed\n");
    delete_module_info(module);
    return 1;
  }

  for (int i = 0; i < batch * n; i++)
  {
    input[i] = 0;
  }  

  // Two simple polynomials with small signed coefficients.
  input[0] = 1;
  input[1] = 2;
  input[2] = 3;
  //input[n + 0] = 7;
  //input[n + 1] = -5;
  //input[n + 2] = 11;

  VEC_ZNX_DFT* dft = new_vec_znx_dft(module, batch);
  VEC_ZNX_BIG* out_big = new_vec_znx_big(module, batch);
  if (!dft || !out_big) {
    fprintf(stderr, "allocation failed\n");
    delete_vec_znx_dft(dft);
    delete_vec_znx_big(out_big);
    free(input);
    delete_module_info(module);
    return 1;
  }

  vec_znx_dft(module, dft, batch, input, batch, n);

  const uint64_t tmp_bytes = vec_znx_idft_tmp_bytes(module);
  uint8_t* tmp = (uint8_t*)malloc(tmp_bytes);
  if (!tmp) {
    fprintf(stderr, "tmp allocation failed\n");
    delete_vec_znx_dft(dft);
    delete_vec_znx_big(out_big);
    free(input);
    delete_module_info(module);
    return 1;
  }

  // Single API call: runtime dispatch picks GPU path when available,
  // otherwise falls back to existing CPU implementation.
  vec_znx_idft(module, out_big, batch, dft, batch, tmp);

  __int128_t* out = (__int128_t*)out_big;
  printf("vec_znx_idft dispatch demo (N=%" PRIu64 ", batch=%" PRIu64 ")\n", n, batch);
  printf("poly0 coeff[0..3]: ");
  print_i128(out[0]);
  printf(" ");
  print_i128(out[1]);
  printf(" ");
  print_i128(out[2]);
  printf(" ");
  print_i128(out[3]);
  printf("\n");

  free(tmp);
  delete_vec_znx_dft(dft);
  delete_vec_znx_big(out_big);
  free(input);
  delete_module_info(module);
  return 0;
}
