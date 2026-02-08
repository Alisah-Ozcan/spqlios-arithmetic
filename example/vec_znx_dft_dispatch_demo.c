#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "spqlios/arithmetic/vec_znx_arithmetic.h"

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
    input[i] = i;
  }  

  // Two simple polynomials with small signed coefficients.
  input[0] = 1;
  input[1] = 1;
  input[2] = 1;
  //input[n + 0] = 1;
  //input[n + 1] = 1;
  //input[n + 2] = 1;

  VEC_ZNX_DFT* dft = new_vec_znx_dft(module, batch);
  if (!dft) {
    fprintf(stderr, "new_vec_znx_dft failed\n");
    free(input);
    delete_module_info(module);
    return 1;
  }

  // Single API call: runtime dispatch picks GPU path when available,
  // otherwise falls back to existing CPU implementation.
  vec_znx_dft(module, dft, batch, input, batch, n);

  uint64_t* out = (uint64_t*)dft;
  printf("vec_znx_dft dispatch demo (N=%" PRIu64 ", batch=%" PRIu64 ")\n", n, batch);
  printf("poly0 coeff0 limbs: %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n", out[0], out[1], out[2], out[3]);
  printf("poly0 coeff1 limbs: %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n", out[4], out[5], out[6], out[7]);
  printf("poly0 coeff2 limbs: %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n", out[8], out[9], out[10], out[11]);
  printf("poly0 coeff3 limbs: %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n", out[12], out[13], out[14], out[15]);

  delete_vec_znx_dft(dft);
  free(input);
  delete_module_info(module);
  return 0;
}
