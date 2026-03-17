#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "spqlios/arithmetic/vec_znx_arithmetic_private.h"

static void print_prefix_i64(const char* label, const int64_t* poly, uint64_t count) {
  printf("%s:", label);
  for (uint64_t i = 0; i < count; i++) {
    printf(" %" PRId64, poly[i]);
  }
  printf("\n");
}

int main(void) {
  const uint64_t n = 1u << 12;
  const uint64_t batch = 1;

  MODULE* module = new_module_info(n, FFT64);
  if (!module) {
    fprintf(stderr, "new_module_info failed\n");
    return 1;
  }
  if (!module->mod.fft64.p_gpu) {
    fprintf(stderr, "FFT64 GPU precomputation is not available for N=%" PRIu64 "\n", n);
    delete_module_info(module);
    return 2;
  }

  int64_t* input = (int64_t*)calloc(batch * n, sizeof(int64_t));
  if (!input) {
    fprintf(stderr, "input allocation failed\n");
    delete_module_info(module);
    return 1;
  }

  input[0] = 1;
  input[1] = -2;
  input[2] = 5;
  input[17] = -3;
  input[128] = 7;
  input[(n >> 1) + 3] = 4;

  VEC_ZNX_DFT* dft = new_vec_znx_dft(module, batch);
  VEC_ZNX_BIG* recovered = new_vec_znx_big(module, batch);
  if (!dft || !recovered) {
    fprintf(stderr, "allocation failed\n");
    delete_vec_znx_dft(dft);
    delete_vec_znx_big(recovered);
    free(input);
    delete_module_info(module);
    return 1;
  }

  vec_znx_dft(module, dft, batch, input, batch, n);
  vec_znx_idft(module, recovered, batch, dft, batch, NULL);

  printf("fft64 FFNT idft dispatch demo (N=%" PRIu64 ")\n", n);
  print_prefix_i64("input[0..15]", input, 16);
  print_prefix_i64("recovered[0..15]", (int64_t*)recovered, 16);

  delete_vec_znx_big(recovered);
  delete_vec_znx_dft(dft);
  free(input);
  delete_module_info(module);
  return 0;
}
