#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "gpuntt/common/modular_arith.cuh"
#include "gpuntt/ntt_merge/ntt_cpu.cuh"
#include "spqlios/gpu/ntt_parameters.cuh"
#include "spqlios/gpu/vec_znx_gpu.cuh"

static void print_u32_vec(const char* label, const uint32_t* v, uint64_t n, uint64_t size) {
  printf("%s\n", label);
  for (uint64_t p = 0; p < size; p++) {
    printf("poly %" PRIu64 ": ", p);
    for (uint64_t i = 0; i < n; i++) {
      printf("%" PRIu32 "%s", v[p * n + i], (i + 1 == n) ? "\n" : " ");
    }
  }
}

static int log2_u64(uint64_t n) {
  int logn = 0;
  uint64_t v = 1;
  while (v < n) {
    v <<= 1;
    logn++;
  }
  return (v == n) ? logn : -1;
}

static void schoolbook_reference_mul(const uint32_t* a, const uint32_t* b, uint32_t* out, uint64_t n,
                                     gpuntt::ReductionPolynomial poly) {
  const int logn = log2_u64(n);
  if (logn < 1) {
    fprintf(stderr, "schoolbook_reference_mul: n must be power of two\n");
    abort();
  }
  gpuntt::NTTParameters<Data32> params(logn, poly);

  std::vector<Data32> va(n);
  std::vector<Data32> vb(n);
  for (uint64_t i = 0; i < n; i++) {
    va[i] = a[i];
    vb[i] = b[i];
  }
  std::vector<Data32> R = gpuntt::schoolbook_poly_multiplication<Data32>(va, vb, params.modulus, poly);
  for (uint64_t i = 0; i < n; i++) {
    out[i] = R[i];
  }
}

int main(void) {
  const uint64_t N = 16;
  const uint64_t size = 1;

  uint32_t a[N * size];
  uint32_t b[N * size];
  for (uint64_t p = 0; p < size; p++) {
    for (uint64_t i = 0; i < N; i++) {
      a[p * N + i] = (uint32_t)(i + 1 + p);
      b[p * N + i] = (uint32_t)(N - i + p);
    }
  }

  NTTParameterModule params(N, gpuntt::ReductionPolynomial::X_N_plus);
  VEC_GPU<uint32_t> a_dft;
  VEC_GPU<uint32_t> b_dft;
  VEC_GPU<uint32_t> c_dft;

  vec_znx_dft_gpu(a_dft, a, params, size);
  vec_znx_dft_gpu(b_dft, b, params, size);
  vec_dft_hadamard_gpu(c_dft, a_dft, b_dft, params, size);

  uint32_t out[N * size];
  vec_znx_idft_gpu(out, c_dft, params, size);

  print_u32_vec("a:", a, N, size);
  print_u32_vec("b:", b, N, size);
  print_u32_vec("c = iNTT(NTT(a) .* NTT(b)):", out, N, size);

  bool ok = true;
  for (uint64_t p = 0; p < size; p++) {
    uint32_t expected[N];
    schoolbook_reference_mul(a + p * N, b + p * N, expected, N, params.poly());
    for (uint64_t i = 0; i < N; i++) {
      uint32_t got = out[p * N + i];
      if (got != expected[i]) {
        fprintf(stderr, "mismatch: poly=%" PRIu64 " idx=%" PRIu64 " got=%" PRIu32 " exp=%" PRIu32 "\n", p, i, got,
                expected[i]);
        ok = false;
        break;
      }
    }
    if (!ok) {
      break;
    }
  }
  printf("check: %s\n", ok ? "OK" : "FAIL");

  return 0;
}
