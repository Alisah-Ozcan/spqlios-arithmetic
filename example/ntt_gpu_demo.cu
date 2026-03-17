#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "spqlios/arithmetic/vec_znx_arithmetic.h"

int main(void) {
  
  const uint64_t N = 1024;
  MODULE* module = new_module_info(N, FFT64);
  if (!module) {
    fprintf(stderr, "failed to create module\n");
    return 1;
  }

  printf("module created: N=%" PRIu64 "\n", module_get_n(module));
  delete_module_info(module);
  

  return 0;
}
