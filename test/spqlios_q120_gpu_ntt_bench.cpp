#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <random>
#include <vector>

#include "spqlios/arithmetic/vec_znx_arithmetic_private.h"
#include "spqlios/q120/q120_common.h"

#if !defined(SPQLIOS_USE_GPU)
#error "This benchmark requires SPQLIOS_USE_GPU=ON"
#endif

namespace {

static void apply_gpu_ntt_args(benchmark::internal::Benchmark* bench) {
  static const int64_t sizes[] = {1 << 10, 1 << 11, 1 << 12, 1 << 13, 1 << 14, 1 << 15, 1 << 16};
  static const int64_t batches[] = {1, 4, 16, 32, 64};
  for (int64_t n : sizes) {
    for (int64_t batch : batches) {
      bench->Args({n, batch});
    }
  }
}

static uint64_t make_seeded_value(std::mt19937_64& rng, uint64_t q) {
  return rng() % q;
}

static bool init_gpu_rns_input(std::vector<uint64_t>& host_rns, uint64_t n, uint64_t batch) {
  host_rns.resize((size_t)(n * batch * 4));
  std::mt19937_64 rng(0xC0FFEEULL);
  for (uint64_t p = 0; p < batch; p++) {
    for (uint64_t m = 0; m < 4; m++) {
      const uint64_t q = (uint64_t)PRIMES_VEC[m];
      for (uint64_t i = 0; i < n; i++) {
        host_rns[((p << 2) + m) * n + i] = make_seeded_value(rng, q);
      }
    }
  }
  return true;
}

static bool prepare_module_and_device_buffer(MODULE*& module, uint64_t*& d_data, uint64_t n, uint64_t batch,
                                             benchmark::State& state) {
  module = new_module_info(n, NTT120);
  if (!module || !module->mod.q120.p_gpu) {
    state.SkipWithError("GPU precomputation is not available for this module");
    return false;
  }

  std::vector<uint64_t> host_rns;
  init_gpu_rns_input(host_rns, n, batch);

  const size_t word_count = (size_t)(n * batch * 4);
  if (cudaMalloc((void**)&d_data, word_count * sizeof(uint64_t)) != cudaSuccess) {
    state.SkipWithError("cudaMalloc failed");
    return false;
  }
  if (cudaMemcpy(d_data, host_rns.data(), word_count * sizeof(uint64_t), cudaMemcpyHostToDevice) != cudaSuccess) {
    state.SkipWithError("cudaMemcpy H2D failed");
    cudaFree(d_data);
    d_data = nullptr;
    return false;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    state.SkipWithError("cudaDeviceSynchronize failed");
    cudaFree(d_data);
    d_data = nullptr;
    return false;
  }
  return true;
}

}  // namespace

static void benchmark_gpu_ntt_inplace(benchmark::State& state) {
  const uint64_t n = (uint64_t)state.range(0);
  const uint64_t batch = (uint64_t)state.range(1);

  MODULE* module = nullptr;
  uint64_t* d_data = nullptr;
  if (!prepare_module_and_device_buffer(module, d_data, n, batch, state)) {
    if (module) {
      delete_module_info(module);
    }
    return;
  }

  for (auto _ : state) {
    if (!q120_vec_gpu_ntt_inplace_raw(module, d_data, batch)) {
      state.SkipWithError("q120_vec_gpu_ntt_inplace_raw failed");
      break;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
      state.SkipWithError("cudaDeviceSynchronize after NTT failed");
      break;
    }

    state.PauseTiming();
    if (!q120_vec_gpu_intt_inplace_raw(module, d_data, batch) || cudaDeviceSynchronize() != cudaSuccess) {
      state.SkipWithError("restore (INTT) failed");
      state.ResumeTiming();
      break;
    }
    state.ResumeTiming();
  }

  state.SetItemsProcessed(state.iterations() * (int64_t)(n * batch));
  cudaFree(d_data);
  delete_module_info(module);
}

static void benchmark_gpu_intt_inplace(benchmark::State& state) {
  const uint64_t n = (uint64_t)state.range(0);
  const uint64_t batch = (uint64_t)state.range(1);

  MODULE* module = nullptr;
  uint64_t* d_data = nullptr;
  if (!prepare_module_and_device_buffer(module, d_data, n, batch, state)) {
    if (module) {
      delete_module_info(module);
    }
    return;
  }

  if (!q120_vec_gpu_ntt_inplace_raw(module, d_data, batch) || cudaDeviceSynchronize() != cudaSuccess) {
    state.SkipWithError("initial NTT preparation failed");
    cudaFree(d_data);
    delete_module_info(module);
    return;
  }

  for (auto _ : state) {
    if (!q120_vec_gpu_intt_inplace_raw(module, d_data, batch)) {
      state.SkipWithError("q120_vec_gpu_intt_inplace_raw failed");
      break;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
      state.SkipWithError("cudaDeviceSynchronize after INTT failed");
      break;
    }

    state.PauseTiming();
    if (!q120_vec_gpu_ntt_inplace_raw(module, d_data, batch) || cudaDeviceSynchronize() != cudaSuccess) {
      state.SkipWithError("restore (NTT) failed");
      state.ResumeTiming();
      break;
    }
    state.ResumeTiming();
  }

  state.SetItemsProcessed(state.iterations() * (int64_t)(n * batch));
  cudaFree(d_data);
  delete_module_info(module);
}

BENCHMARK(benchmark_gpu_ntt_inplace)->Name("vec_gpu_ntt_inplace")->Apply(apply_gpu_ntt_args);
BENCHMARK(benchmark_gpu_intt_inplace)->Name("vec_gpu_intt_inplace")->Apply(apply_gpu_ntt_args);

BENCHMARK_MAIN();
