#include <cuda_runtime.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#include "gpuntt/common/modular_arith.cuh"
#include "gpuntt/ntt_merge/ntt.cuh"
#include "vec_znx_gpu.cuh"

static void require_size(const VEC_GPU<uint32_t>& v, uint64_t expected, const char* fn) {
  if (v.size() != expected) {
    fprintf(stderr, "%s: size mismatch (have=%zu expected=%" PRIu64 ")\n", fn, v.size(), expected);
    abort();
  }
}

void vec_znx_dft_gpu(VEC_GPU<uint32_t>& out, const uint32_t* host_in, const NTTParameterModule& params,
                     uint64_t batch) {
  if (!host_in || batch == 0) {
    fprintf(stderr, "vec_znx_dft_gpu: invalid input\n");
    abort();
  }
  const uint64_t n = params.n();
  const uint64_t total = n * batch;
  out.copy_from_host(host_in, total);
  gpuntt::ntt_configuration<Data32> cfg = {
      .n_power = params.logn(),
      .ntt_type = gpuntt::FORWARD,
      .ntt_layout = gpuntt::PerPolynomial,
      .reduction_poly = params.poly(),
      .zero_padding = false,
      .mod_inverse = params.n_inv(),
      .stream = 0,
  };
  gpuntt::GPU_NTT_Inplace<Data32>(out.data(), params.forward_table(), params.modulus(), cfg, (int)batch);
}

void vec_znx_dft_gpu(VEC_GPU<uint32_t>& out, const VEC_GPU<uint32_t>& in, const NTTParameterModule& params,
                     uint64_t batch) {
  if (batch == 0) {
    fprintf(stderr, "vec_znx_dft_gpu: invalid input\n");
    abort();
  }
  const uint64_t n = params.n();
  const uint64_t total = n * batch;
  require_size(in, total, "vec_znx_dft_gpu");
  if (out.size() != total) {
    out.allocate(total);
  }
  if (out.data() != in.data()) {
    spqlios_cuda_check(cudaMemcpy(out.data(), in.data(), total * sizeof(uint32_t), cudaMemcpyDeviceToDevice),
                       "cudaMemcpy D2D");
  }
  gpuntt::ntt_configuration<Data32> cfg = {
      .n_power = params.logn(),
      .ntt_type = gpuntt::FORWARD,
      .ntt_layout = gpuntt::PerPolynomial,
      .reduction_poly = params.poly(),
      .zero_padding = false,
      .mod_inverse = params.n_inv(),
      .stream = 0,
  };
  gpuntt::GPU_NTT_Inplace<Data32>(out.data(), params.forward_table(), params.modulus(), cfg, (int)batch);
}

__global__ static void hadamard_kernel(uint32_t* out, const uint32_t* a, const uint32_t* b, uint64_t total,
                                       Modulus32 mod) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) {
    out[i] = OPERATOR_GPU_32::mult(a[i], b[i], mod);
  }
}

void vec_dft_hadamard_gpu(VEC_GPU<uint32_t>& out, const VEC_GPU<uint32_t>& a, const VEC_GPU<uint32_t>& b,
                          const NTTParameterModule& params, uint64_t batch) {
  if (batch == 0) {
    fprintf(stderr, "vec_dft_hadamard_gpu: invalid input\n");
    abort();
  }
  const uint64_t n = params.n();
  const uint64_t total = n * batch;
  require_size(a, total, "vec_dft_hadamard_gpu");
  require_size(b, total, "vec_dft_hadamard_gpu");
  if (out.size() != total) {
    out.allocate(total);
  }
  const dim3 block(256);
  const dim3 grid((total + block.x - 1) / block.x);
  hadamard_kernel<<<grid, block>>>(out.data(), a.data(), b.data(), total, params.modulus());
  spqlios_cuda_check(cudaGetLastError(), "hadamard_kernel launch");
}

void vec_znx_idft_gpu(uint32_t* host_out, VEC_GPU<uint32_t>& in, const NTTParameterModule& params, uint64_t batch) {
  if (!host_out || batch == 0) {
    fprintf(stderr, "vec_znx_idft_gpu: invalid input\n");
    abort();
  }
  const uint64_t n = params.n();
  const uint64_t total = n * batch;
  require_size(in, total, "vec_znx_idft_gpu");
  gpuntt::ntt_configuration<Data32> cfg = {
      .n_power = params.logn(),
      .ntt_type = gpuntt::INVERSE,
      .ntt_layout = gpuntt::PerPolynomial,
      .reduction_poly = params.poly(),
      .zero_padding = false,
      .mod_inverse = params.n_inv(),
      .stream = 0,
  };
  gpuntt::GPU_INTT_Inplace<Data32>(in.data(), params.inverse_table(), params.modulus(), cfg, (int)batch);
  in.copy_to_host(host_out, total);
}
