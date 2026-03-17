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

  int64_t* a = (int64_t*)calloc(n, sizeof(int64_t));
  int64_t* b = (int64_t*)calloc(n, sizeof(int64_t));
  int64_t* gpu_res = (int64_t*)calloc(n, sizeof(int64_t));
  int64_t* cpu_res = (int64_t*)calloc(n, sizeof(int64_t));
  uint8_t* tmp = (uint8_t*)malloc(znx_small_single_product_tmp_bytes(module));
  if (!a || !b || !gpu_res || !cpu_res || !tmp) {
    fprintf(stderr, "allocation failed\n");
    free(tmp);
    free(cpu_res);
    free(gpu_res);
    free(b);
    free(a);
    delete_module_info(module);
    return 1;
  }

  a[0] = 1;
  a[1] = 2;
  a[7] = -1;
  a[33] = 4;
  a[(n >> 1) + 2] = -3;

  b[0] = -2;
  b[2] = 1;
  b[9] = 5;
  b[17] = -4;
  b[(n >> 1) + 1] = 2;

  if (!fft64_znx_small_single_product_gpu(module, gpu_res, a, b)) {
    fprintf(stderr, "fft64_znx_small_single_product_gpu failed\n");
    free(tmp);
    free(cpu_res);
    free(gpu_res);
    free(b);
    free(a);
    delete_module_info(module);
    return 1;
  }

  znx_small_single_product(module, cpu_res, a, b, tmp);

  printf("fft64 FFNT gpu poly multiplication demo (N=%" PRIu64 ")\n", n);
  print_prefix_i64("a[0..15]", a, 16);
  print_prefix_i64("b[0..15]", b, 16);
  print_prefix_i64("gpu[0..15]", gpu_res, 16);
  print_prefix_i64("cpu[0..15]", cpu_res, 16);

  free(tmp);
  free(cpu_res);
  free(gpu_res);
  free(b);
  free(a);
  delete_module_info(module);
  return 0;
}
