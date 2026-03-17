#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "spqlios/arithmetic/vec_znx_arithmetic_private.h"

int main(void) {
  const uint64_t n = 1u << 12;
  const uint64_t batch = 1;
  const uint64_t m = n >> 1;

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
  input[1] = 2;
  input[2] = 3;
  input[3] = 4;
  input[m + 0] = -1;
  input[m + 1] = 1;

  VEC_ZNX_DFT* dft = new_vec_znx_dft(module, batch);
  if (!dft) {
    fprintf(stderr, "new_vec_znx_dft failed\n");
    free(input);
    delete_module_info(module);
    return 1;
  }

  vec_znx_dft(module, dft, batch, input, batch, n);

  printf("fft64 FFNT dispatch demo (N=%" PRIu64 ", m=%" PRIu64 ")\n", n, m);
  double* out = (double*)dft;
  for (uint64_t i = 0; i < 4; i++) {
    printf("  idx %" PRIu64 ": %.6f + %.6fi\n", i, out[i], out[m + i]);
  }

  delete_vec_znx_dft(dft);
  free(input);
  delete_module_info(module);
  return 0;
}
