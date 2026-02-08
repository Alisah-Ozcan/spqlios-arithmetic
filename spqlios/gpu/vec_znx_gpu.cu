#include <cuda_runtime.h>
#include <exception>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "gpuntt/common/modular_arith.cuh"
#include "gpuntt/ntt_merge/ntt.cuh"
#include "vec_znx_gpu.cuh"

extern "C" {
#include "../arithmetic/vec_znx_arithmetic_private.h"
#include "../q120/q120_arithmetic.h"
#include "../q120/q120_common.h"
}

int log2_u64(uint64_t n) {
  int logn = 0;
  uint64_t v = 1;
  while (v < n) {
    v <<= 1;
    logn++;
  }
  return (v == n) ? logn : -1;
}

uint64_t mul_mod_u64(uint64_t a, uint64_t b, uint64_t mod) {
  return (uint64_t)((__uint128_t)a * (__uint128_t)b % (__uint128_t)mod);
}

uint64_t modq_pow_u64(uint64_t x, uint64_t e, uint64_t q) {
  uint64_t res = 1;
  uint64_t base = x % q;
  uint64_t exp = e;
  while (exp) {
    if (exp & 1) {
      res = mul_mod_u64(res, base, q);
    }
    base = mul_mod_u64(base, base, q);
    exp >>= 1;
  }
  return res;
}

__global__ static void znx64_to_packed_rns_kernel(Data64* dst, const int64_t* src, uint64_t nn, uint64_t poly_count,
                                                  uint64_t src_stride, uint64_t oq0, uint64_t oq1, uint64_t oq2,
                                                  uint64_t oq3, uint64_t q0, uint64_t q1, uint64_t q2,
                                                  uint64_t q3) {
  const uint64_t coeff = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t mod = (uint64_t)blockIdx.y;
  const uint64_t poly = (uint64_t)blockIdx.z;
  if (coeff >= nn || mod >= 4 || poly >= poly_count) {
    return;
  }

  const uint64_t x = (uint64_t)src[poly * src_stride + coeff];
  const uint64_t x_lo = x & UINT64_C(0x7fffffffffffffff);
  const uint64_t x_hi = x & UINT64_C(0x8000000000000000);
  const uint64_t oq = (mod == 0) ? oq0 : (mod == 1) ? oq1 : (mod == 2) ? oq2 : oq3;
  const uint64_t q = (mod == 0) ? q0 : (mod == 1) ? q1 : (mod == 2) ? q2 : q3;
  const uint64_t dst_idx = ((poly << 2) + mod) * nn + coeff;
  // GPUNTT path expects residues in [0, q). Canonicalize here.
  dst[dst_idx] = (Data64)((x_lo + (x_hi ? oq : 0)) % q);
}

static constexpr uint32_t PACKED_TRANSPOSE_TILE_COEFF = 128;
static constexpr __uint128_t Q120_MOD = (__uint128_t)Q1 * Q2 * Q3 * Q4;
static constexpr __uint128_t Q120_HALF_UP = (Q120_MOD + 1) / 2;
static constexpr __uint128_t Q120_QM1 = (__uint128_t)Q2 * Q3 * Q4;
static constexpr __uint128_t Q120_QM2 = (__uint128_t)Q1 * Q3 * Q4;
static constexpr __uint128_t Q120_QM3 = (__uint128_t)Q1 * Q2 * Q4;
static constexpr __uint128_t Q120_QM4 = (__uint128_t)Q1 * Q2 * Q3;
static constexpr uint64_t Q120_MOD_LO = (uint64_t)Q120_MOD;
static constexpr uint64_t Q120_MOD_HI = (uint64_t)(Q120_MOD >> 64);
static constexpr uint64_t Q120_HALF_UP_LO = (uint64_t)Q120_HALF_UP;
static constexpr uint64_t Q120_HALF_UP_HI = (uint64_t)(Q120_HALF_UP >> 64);
static constexpr uint64_t Q120_QM1_LO = (uint64_t)Q120_QM1;
static constexpr uint64_t Q120_QM1_HI = (uint64_t)(Q120_QM1 >> 64);
static constexpr uint64_t Q120_QM2_LO = (uint64_t)Q120_QM2;
static constexpr uint64_t Q120_QM2_HI = (uint64_t)(Q120_QM2 >> 64);
static constexpr uint64_t Q120_QM3_LO = (uint64_t)Q120_QM3;
static constexpr uint64_t Q120_QM3_HI = (uint64_t)(Q120_QM3 >> 64);
static constexpr uint64_t Q120_QM4_LO = (uint64_t)Q120_QM4;
static constexpr uint64_t Q120_QM4_HI = (uint64_t)(Q120_QM4 >> 64);

static bool is_device_accessible_ptr(const void* ptr) {
  if (!ptr) {
    return false;
  }
  cudaPointerAttributes attr;
  const cudaError_t st = cudaPointerGetAttributes(&attr, ptr);
  if (st == cudaSuccess) {
#if CUDART_VERSION >= 10000
    return attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged;
#else
    return attr.memoryType == cudaMemoryTypeDevice;
#endif
  }
  if (st == cudaErrorInvalidValue) {
    (void)cudaGetLastError();
    return false;
  }
  spqlios_cuda_check(st, "cudaPointerGetAttributes");
  return false;
}

static void vec_gpu_ntt_inplace(Data64* output, const Data64* input, const MODULE* module, uint64_t batch_size) {
  if (!output || !input || !module || module->module_type != NTT120 || !module->mod.q120.p_gpu ||
      module->mod.q120.p_gpu->mod_count <= 0) {
    throw std::runtime_error("vec_gpu_ntt_inplace: invalid input/module");
  }
  const q120_gpu_module_info_t* gpu = module->mod.q120.p_gpu;
  const int poly_count = (int)(batch_size * (uint64_t)gpu->mod_count);
  const gpuntt::ntt_rns_configuration<Data64> cfg = {
      .n_power = gpu->logn,
      .ntt_type = gpuntt::FORWARD,
      .ntt_layout = gpuntt::PerPolynomial,
      .reduction_poly = gpuntt::X_N_plus,
      .zero_padding = false,
      .mod_inverse = nullptr,
      .stream = 0,
  };

  const uint64_t n = module->nn;
  const uint64_t elem_count = n * (uint64_t)poly_count;
  const size_t bytes = (size_t)elem_count * sizeof(Data64);
  const bool input_on_device = is_device_accessible_ptr(input);
  const bool output_on_device = is_device_accessible_ptr(output);

  if (input_on_device && output_on_device) {
    if (output != input) {
      spqlios_cuda_check(cudaMemcpy(output, input, bytes, cudaMemcpyDeviceToDevice), "cudaMemcpy D2D (NTT in->out)");
    }
    gpuntt::GPU_NTT_Inplace<Data64>(output, gpu->ntt_roots.data(), gpu->moduli.data(), cfg, poly_count, gpu->mod_count);
    return;
  }

  VEC_GPU<Data64> staging(elem_count);
  if (input_on_device) {
    spqlios_cuda_check(cudaMemcpy(staging.data(), input, bytes, cudaMemcpyDeviceToDevice),
                       "cudaMemcpy D2D (NTT in->staging)");
  } else {
    spqlios_cuda_check(cudaMemcpy(staging.data(), input, bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy H2D (NTT in->staging)");
  }

  gpuntt::GPU_NTT_Inplace<Data64>(staging.data(), gpu->ntt_roots.data(), gpu->moduli.data(), cfg, poly_count,
                                  gpu->mod_count);

  if (output_on_device) {
    spqlios_cuda_check(cudaMemcpy(output, staging.data(), bytes, cudaMemcpyDeviceToDevice),
                       "cudaMemcpy D2D (NTT staging->out)");
  } else {
    spqlios_cuda_check(cudaMemcpy(output, staging.data(), bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy D2H (NTT staging->out)");
  }
}

static void vec_gpu_intt_inplace(Data64* output, const Data64* input, const MODULE* module, uint64_t batch_size) {
  if (!output || !input || !module || module->module_type != NTT120 || !module->mod.q120.p_gpu ||
      module->mod.q120.p_gpu->mod_count <= 0) {
    throw std::runtime_error("vec_gpu_intt_inplace: invalid input/module");
  }
  const q120_gpu_module_info_t* gpu = module->mod.q120.p_gpu;
  const int poly_count = (int)(batch_size * (uint64_t)gpu->mod_count);
  const gpuntt::ntt_rns_configuration<Data64> cfg = {
      .n_power = gpu->logn,
      .ntt_type = gpuntt::INVERSE,
      .ntt_layout = gpuntt::PerPolynomial,
      .reduction_poly = gpuntt::X_N_plus,
      .zero_padding = false,
      .mod_inverse = gpu->n_inv.data(),
      .stream = 0,
  };

  const uint64_t n = module->nn;
  const uint64_t elem_count = n * (uint64_t)poly_count;
  const size_t bytes = (size_t)elem_count * sizeof(Data64);
  const bool input_on_device = is_device_accessible_ptr(input);
  const bool output_on_device = is_device_accessible_ptr(output);

  if (input_on_device && output_on_device) {
    if (output != input) {
      spqlios_cuda_check(cudaMemcpy(output, input, bytes, cudaMemcpyDeviceToDevice), "cudaMemcpy D2D (INTT in->out)");
    }
    gpuntt::GPU_INTT_Inplace<Data64>(output, gpu->intt_roots.data(), gpu->moduli.data(), cfg, poly_count,
                                     gpu->mod_count);
    return;
  }

  VEC_GPU<Data64> staging(elem_count);
  if (input_on_device) {
    spqlios_cuda_check(cudaMemcpy(staging.data(), input, bytes, cudaMemcpyDeviceToDevice),
                       "cudaMemcpy D2D (INTT in->staging)");
  } else {
    spqlios_cuda_check(cudaMemcpy(staging.data(), input, bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy H2D (INTT in->staging)");
  }

  gpuntt::GPU_INTT_Inplace<Data64>(staging.data(), gpu->intt_roots.data(), gpu->moduli.data(), cfg, poly_count,
                                   gpu->mod_count);

  if (output_on_device) {
    spqlios_cuda_check(cudaMemcpy(output, staging.data(), bytes, cudaMemcpyDeviceToDevice),
                       "cudaMemcpy D2D (INTT staging->out)");
  } else {
    spqlios_cuda_check(cudaMemcpy(output, staging.data(), bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy D2H (INTT staging->out)");
  }
}

__global__ static void packed_rns_to_spqlios_kernel(uint64_t* dst, const Data64* src, uint64_t nn, uint64_t poly_count) {
  const uint64_t poly = (uint64_t)blockIdx.z;
  if (poly >= poly_count) {
    return;
  }
  const uint64_t coeff_base = (uint64_t)blockIdx.x * PACKED_TRANSPOSE_TILE_COEFF;
  const uint32_t tid = threadIdx.x;
  __shared__ Data64 tile[4][PACKED_TRANSPOSE_TILE_COEFF + 1];

  for (uint32_t t = tid; t < 4u * PACKED_TRANSPOSE_TILE_COEFF; t += blockDim.x) {
    const uint32_t mod = t / PACKED_TRANSPOSE_TILE_COEFF;
    const uint32_t off = t - mod * PACKED_TRANSPOSE_TILE_COEFF;
    const uint64_t coeff = coeff_base + off;
    if (coeff < nn) {
      tile[mod][off] = src[((poly << 2) + mod) * nn + coeff];
    }
  }
  __syncthreads();
  for (uint32_t t = tid; t < 4u * PACKED_TRANSPOSE_TILE_COEFF; t += blockDim.x) {
    const uint32_t off = t >> 2;
    const uint32_t mod = t & 3u;
    const uint64_t coeff = coeff_base + off;
    if (coeff < nn) {
      dst[(poly * nn + coeff) * 4 + mod] = (uint64_t)tile[mod][off];
    }
  }
}

__global__ static void spqlios_to_packed_rns_kernel(Data64* dst, const uint64_t* src, uint64_t nn, uint64_t poly_count,
                                                    uint64_t q0, uint64_t q1, uint64_t q2, uint64_t q3) {
  const uint64_t poly = (uint64_t)blockIdx.z;
  if (poly >= poly_count) {
    return;
  }
  const uint64_t coeff_base = (uint64_t)blockIdx.x * PACKED_TRANSPOSE_TILE_COEFF;
  const uint32_t tid = threadIdx.x;
  __shared__ Data64 tile[4][PACKED_TRANSPOSE_TILE_COEFF + 1];

  for (uint32_t t = tid; t < 4u * PACKED_TRANSPOSE_TILE_COEFF; t += blockDim.x) {
    const uint32_t off = t >> 2;
    const uint32_t mod = t & 3u;
    const uint64_t coeff = coeff_base + off;
    if (coeff < nn) {
      const uint64_t q = (mod == 0) ? q0 : (mod == 1) ? q1 : (mod == 2) ? q2 : q3;
      // Accept non-canonical q120b limbs from CPU/GPU producers.
      tile[mod][off] = (Data64)(src[(poly * nn + coeff) * 4 + mod] % q);
    }
  }
  __syncthreads();
  for (uint32_t t = tid; t < 4u * PACKED_TRANSPOSE_TILE_COEFF; t += blockDim.x) {
    const uint32_t mod = t / PACKED_TRANSPOSE_TILE_COEFF;
    const uint32_t off = t - mod * PACKED_TRANSPOSE_TILE_COEFF;
    const uint64_t coeff = coeff_base + off;
    if (coeff < nn) {
      dst[((poly << 2) + mod) * nn + coeff] = tile[mod][off];
    }
  }
}

__device__ __forceinline__ static bool ge_u128(uint64_t a_hi, uint64_t a_lo, uint64_t b_hi, uint64_t b_lo) {
  return (a_hi > b_hi) || (a_hi == b_hi && a_lo >= b_lo);
}

__device__ __forceinline__ static void sub_u128_inplace(uint64_t& a_lo, uint64_t& a_hi, uint64_t b_lo, uint64_t b_hi) {
  const uint64_t borrow = (a_lo < b_lo);
  a_lo -= b_lo;
  a_hi = a_hi - b_hi - borrow;
}

__device__ __forceinline__ static void add_small_mul_u128_inplace(uint64_t& acc_lo, uint64_t& acc_hi, uint64_t a,
                                                                   uint64_t b_lo, uint64_t b_hi) {
  const uint64_t prod_lo = a * b_lo;
  const uint64_t prod_hi = a * b_hi + __umul64hi(a, b_lo);
  const uint64_t old_lo = acc_lo;
  acc_lo += prod_lo;
  acc_hi += prod_hi + (acc_lo < old_lo ? 1u : 0u);
}

__global__ static void packed_rns_to_znx128_kernel(ulonglong2* dst, const Data64* src, uint64_t nn, uint64_t poly_count) {
  const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t total = nn * poly_count;
  if (idx >= total) {
    return;
  }
  const uint64_t poly = idx / nn;
  const uint64_t coeff = idx - poly * nn;

  const uint64_t x1 = (uint64_t)src[((poly << 2) + 0) * nn + coeff] % (uint64_t)Q1;
  const uint64_t x2 = (uint64_t)src[((poly << 2) + 1) * nn + coeff] % (uint64_t)Q2;
  const uint64_t x3 = (uint64_t)src[((poly << 2) + 2) * nn + coeff] % (uint64_t)Q3;
  const uint64_t x4 = (uint64_t)src[((poly << 2) + 3) * nn + coeff] % (uint64_t)Q4;

  const uint64_t t1 = (x1 * (uint64_t)Q1_CRT_CST) % (uint64_t)Q1;
  const uint64_t t2 = (x2 * (uint64_t)Q2_CRT_CST) % (uint64_t)Q2;
  const uint64_t t3 = (x3 * (uint64_t)Q3_CRT_CST) % (uint64_t)Q3;
  const uint64_t t4 = (x4 * (uint64_t)Q4_CRT_CST) % (uint64_t)Q4;

  uint64_t acc_lo = 0;
  uint64_t acc_hi = 0;
  add_small_mul_u128_inplace(acc_lo, acc_hi, t1, Q120_QM1_LO, Q120_QM1_HI);
  add_small_mul_u128_inplace(acc_lo, acc_hi, t2, Q120_QM2_LO, Q120_QM2_HI);
  add_small_mul_u128_inplace(acc_lo, acc_hi, t3, Q120_QM3_LO, Q120_QM3_HI);
  add_small_mul_u128_inplace(acc_lo, acc_hi, t4, Q120_QM4_LO, Q120_QM4_HI);

#pragma unroll
  for (int i = 0; i < 3; i++) {
    if (ge_u128(acc_hi, acc_lo, Q120_MOD_HI, Q120_MOD_LO)) {
      sub_u128_inplace(acc_lo, acc_hi, Q120_MOD_LO, Q120_MOD_HI);
    }
  }

  if (ge_u128(acc_hi, acc_lo, Q120_HALF_UP_HI, Q120_HALF_UP_LO)) {
    sub_u128_inplace(acc_lo, acc_hi, Q120_MOD_LO, Q120_MOD_HI);
  }

  dst[idx] = make_ulonglong2(acc_lo, acc_hi);
}

extern "C" EXPORT struct q120_gpu_module_info_t* q120_new_ntt_gpu_precomp(uint64_t n) {
  try {
    if (n == 0 || (n & (n - 1)) != 0 || n > (UINT64_C(1) << 16)) {
      fprintf(stderr, "q120_new_ntt_gpu_precomp: invalid n\n");
      return nullptr;
    }
    const int logn = log2_u64(n);
    if (logn < 1) {
      fprintf(stderr, "q120_new_ntt_gpu_precomp: n must be power of two\n");
      return nullptr;
    }

    const int mod_count = 4;
    auto* precomp = new q120_gpu_module_info_t();
    precomp->n = n;
    precomp->logn = logn;
    precomp->mod_count = mod_count;

    std::vector<Modulus64> moduli(mod_count);
    std::vector<Ninverse64> n_inv(mod_count);
    std::vector<Root64> forward_roots((size_t)mod_count * n);
    std::vector<Root64> inverse_roots((size_t)mod_count * n);

    const uint64_t exp = (UINT64_C(1) << 16) / n;
    for (int k = 0; k < mod_count; k++) {
      const uint64_t q = (uint64_t)PRIMES_VEC[k];
      const uint64_t psi = modq_pow_u64((uint64_t)OMEGAS_VEC[k], exp, q);
      const uint64_t omega = mul_mod_u64(psi, psi, q);
      precomp->oq[k] = q - (UINT64_C(0x8000000000000000) % q);

      gpuntt::NTTFactors<Data64> factors(Modulus64(q), (Data64)omega, (Data64)psi);
      gpuntt::NTTParameters<Data64> params(logn, factors, gpuntt::ReductionPolynomial::X_N_plus);

      moduli[k] = params.modulus;
      n_inv[k] = params.n_inv;

      std::vector<Root<Data64>> fwd = params.gpu_root_of_unity_table_generator(params.forward_root_of_unity_table);
      std::vector<Root<Data64>> inv = params.gpu_root_of_unity_table_generator(params.inverse_root_of_unity_table);
      if (fwd.size() != n || inv.size() != n) {
        fprintf(stderr, "q120_new_ntt_gpu_precomp: root table size mismatch\n");
        return nullptr;
      }
      for (uint64_t i = 0; i < n; i++) {
        forward_roots[(size_t)k * n + i] = (Root64)fwd[i];
        inverse_roots[(size_t)k * n + i] = (Root64)inv[i];
      }
    }

    precomp->moduli.copy_from_host(moduli.data(), moduli.size());
    precomp->n_inv.copy_from_host(n_inv.data(), n_inv.size());
    precomp->ntt_roots.copy_from_host(forward_roots.data(), forward_roots.size());
    precomp->intt_roots.copy_from_host(inverse_roots.data(), inverse_roots.size());

    return precomp;
  } catch (const std::exception& e) {
    fprintf(stderr, "q120_new_ntt_gpu_precomp: GPU initialization failed: %s\n", e.what());
    return nullptr;
  } catch (...) {
    fprintf(stderr, "q120_new_ntt_gpu_precomp: GPU initialization failed with unknown error\n");
    return nullptr;
  }
}

extern "C" EXPORT void q120_del_ntt_gpu_precomp(struct q120_gpu_module_info_t* precomp) { delete precomp; }

extern "C" EXPORT int q120_vec_znx_dft_gpu(const MODULE* module, VEC_ZNX_DFT* res, uint64_t res_size,
                                           const int64_t* a, uint64_t a_size, uint64_t a_sl) {
  try {
    if (!module || !res || !a || module->module_type != NTT120) {
      return 0;
    }
    auto* gpu = module->mod.q120.p_gpu;
    if (!gpu || gpu->mod_count <= 0) {
      return 0;
    }
    if (gpu->mod_count != 4) {
      return 0;
    }

    const uint64_t nn = module->nn;
    const uint64_t smin = res_size < a_size ? res_size : a_size;
    uint64_t* const tres = (uint64_t*)res;

    // Keep API behavior identical with CPU path for tail entries only.
    if (res_size > smin) {
      memset(tres + smin * nn * 4, 0, (res_size - smin) * nn * 4 * sizeof(uint64_t));
    }
    if (smin == 0) {
      return 1;
    }

    const uint64_t total_rns = smin * nn * 4;
    VEC_GPU<Data64> device_rns(total_rns);
    VEC_GPU<uint64_t> device_spqlios(total_rns);
    VEC_GPU<int64_t> device_input((size_t)smin * a_sl);
    device_input.copy_from_host(a, (size_t)smin * a_sl);

    const dim3 block_in(512);
    const dim3 grid_in((nn + block_in.x - 1) / block_in.x, 4, smin);
    znx64_to_packed_rns_kernel<<<grid_in, block_in>>>(device_rns.data(), device_input.data(), nn, smin, a_sl,
                                                      gpu->oq[0], gpu->oq[1], gpu->oq[2], gpu->oq[3], (uint64_t)Q1,
                                                      (uint64_t)Q2, (uint64_t)Q3, (uint64_t)Q4);
    spqlios_cuda_check(cudaGetLastError(), "znx64_to_packed_rns_kernel launch");

    vec_gpu_ntt_inplace(device_rns.data(), device_rns.data(), module, smin);

    const dim3 block_out(256);
    const dim3 grid_out((nn + PACKED_TRANSPOSE_TILE_COEFF - 1) / PACKED_TRANSPOSE_TILE_COEFF, 1, smin);
    packed_rns_to_spqlios_kernel<<<grid_out, block_out>>>(device_spqlios.data(), device_rns.data(), nn, smin);
    spqlios_cuda_check(cudaGetLastError(), "packed_rns_to_spqlios_kernel launch");
    device_spqlios.copy_to_host(tres, total_rns);

    return 1;
  } catch (const std::exception& e) {
    fprintf(stderr, "q120_vec_znx_dft_gpu: fallback to CPU due to: %s\n", e.what());
    if (strstr(e.what(), "invalid device function") != nullptr) {
      int dev = -1;
      if (cudaGetDevice(&dev) == cudaSuccess) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
          fprintf(stderr, "q120_vec_znx_dft_gpu: runtime device=%d (%s), compute capability=%d.%d\n", dev, prop.name,
                  prop.major, prop.minor);
        }
      }
      int drv = 0;
      int rt = 0;
      if (cudaDriverGetVersion(&drv) == cudaSuccess && cudaRuntimeGetVersion(&rt) == cudaSuccess) {
        fprintf(stderr, "q120_vec_znx_dft_gpu: CUDA versions driver=%d runtime=%d\n", drv, rt);
      }
      fprintf(stderr,
              "q120_vec_znx_dft_gpu: hint: CUDA arch mismatch. Reconfigure with matching -DSPQLIOS_CUDA_ARCH "
              "(e.g. native or your SM version).\n");
    }
    return 0;
  } catch (...) {
    fprintf(stderr, "q120_vec_znx_dft_gpu: fallback to CPU due to unknown error\n");
    return 0;
  }
}

extern "C" EXPORT int q120_vec_znx_idft_gpu(const MODULE* module, VEC_ZNX_BIG* res, uint64_t res_size,
                                            const VEC_ZNX_DFT* a_dft, uint64_t a_size) {
  try {
    if (!module || !res || !a_dft || module->module_type != NTT120) {
      return 0;
    }
    auto* gpu = module->mod.q120.p_gpu;
    if (!gpu || gpu->mod_count <= 0) {
      return 0;
    }
    if (gpu->mod_count != 4) {
      return 0;
    }

    const uint64_t nn = module->nn;
    const uint64_t smin = res_size < a_size ? res_size : a_size;
    __int128_t* const tres = (__int128_t*)res;

    // Keep API behavior identical with CPU path for tail entries only.
    if (res_size > smin) {
      memset(tres + smin * nn, 0, (res_size - smin) * nn * sizeof(*tres));
    }
    if (smin == 0) {
      return 1;
    }

    const uint64_t* const src_dft = (const uint64_t*)a_dft;
    const uint64_t total_rns = smin * nn * 4;
    const uint64_t total_big = smin * nn;
    VEC_GPU<Data64> device_rns(total_rns);
    VEC_GPU<uint64_t> device_spqlios(total_rns);
    device_spqlios.copy_from_host(src_dft, total_rns);

    const dim3 block(256);
    const dim3 grid((nn + PACKED_TRANSPOSE_TILE_COEFF - 1) / PACKED_TRANSPOSE_TILE_COEFF, 1, smin);
    spqlios_to_packed_rns_kernel<<<grid, block>>>(device_rns.data(), device_spqlios.data(), nn, smin, (uint64_t)Q1,
                                                  (uint64_t)Q2, (uint64_t)Q3, (uint64_t)Q4);
    spqlios_cuda_check(cudaGetLastError(), "spqlios_to_packed_rns_kernel launch");

    vec_gpu_intt_inplace(device_rns.data(), device_rns.data(), module, smin);

    VEC_GPU<ulonglong2> device_big(total_big);
    const uint32_t block_big = 256;
    const uint32_t grid_big = (uint32_t)((total_big + block_big - 1) / block_big);
    packed_rns_to_znx128_kernel<<<grid_big, block_big>>>(device_big.data(), device_rns.data(), nn, smin);
    spqlios_cuda_check(cudaGetLastError(), "packed_rns_to_znx128_kernel (IDFT) launch");

    if (is_device_accessible_ptr(tres)) {
      spqlios_cuda_check(cudaMemcpy((void*)tres, (const void*)device_big.data(), total_big * sizeof(ulonglong2),
                                    cudaMemcpyDeviceToDevice),
                         "cudaMemcpy D2D (IDFT output)");
    } else {
      device_big.copy_to_host(reinterpret_cast<ulonglong2*>(tres), total_big);
    }

    return 1;
  } catch (const std::exception& e) {
    fprintf(stderr, "q120_vec_znx_idft_gpu: fallback to CPU due to: %s\n", e.what());
    if (strstr(e.what(), "invalid device function") != nullptr) {
      int dev = -1;
      if (cudaGetDevice(&dev) == cudaSuccess) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
          fprintf(stderr, "q120_vec_znx_idft_gpu: runtime device=%d (%s), compute capability=%d.%d\n", dev, prop.name,
                  prop.major, prop.minor);
        }
      }
      int drv = 0;
      int rt = 0;
      if (cudaDriverGetVersion(&drv) == cudaSuccess && cudaRuntimeGetVersion(&rt) == cudaSuccess) {
        fprintf(stderr, "q120_vec_znx_idft_gpu: CUDA versions driver=%d runtime=%d\n", drv, rt);
      }
      fprintf(stderr,
              "q120_vec_znx_idft_gpu: hint: CUDA arch mismatch. Reconfigure with matching -DSPQLIOS_CUDA_ARCH "
              "(e.g. native or your SM version).\n");
    }
    return 0;
  } catch (...) {
    fprintf(stderr, "q120_vec_znx_idft_gpu: fallback to CPU due to unknown error\n");
    return 0;
  }
}

extern "C" EXPORT int q120_vec_gpu_ntt_inplace_raw(const MODULE* module, uint64_t* inout_rns, uint64_t batch_size) {
  try {
    if (!module || !inout_rns || module->module_type != NTT120 || !module->mod.q120.p_gpu) {
      return 0;
    }
    vec_gpu_ntt_inplace(reinterpret_cast<Data64*>(inout_rns), reinterpret_cast<const Data64*>(inout_rns), module,
                        batch_size);
    return 1;
  } catch (...) {
    return 0;
  }
}

extern "C" EXPORT int q120_vec_gpu_intt_inplace_raw(const MODULE* module, uint64_t* inout_rns, uint64_t batch_size) {
  try {
    if (!module || !inout_rns || module->module_type != NTT120 || !module->mod.q120.p_gpu) {
      return 0;
    }
    vec_gpu_intt_inplace(reinterpret_cast<Data64*>(inout_rns), reinterpret_cast<const Data64*>(inout_rns), module,
                         batch_size);
    return 1;
  } catch (...) {
    return 0;
  }
}
