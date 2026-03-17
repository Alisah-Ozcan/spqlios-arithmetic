#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <random>
#include <vector>

#include "spqlios/arithmetic/vec_znx_arithmetic_private.h"

#if !defined(SPQLIOS_USE_GPU)
#error "This benchmark requires SPQLIOS_USE_GPU=ON"
#endif

namespace {

static void apply_gpu_ffnt_args(benchmark::internal::Benchmark* bench) {
  static const int64_t sizes[] = {1 << 12, 1 << 13, 1 << 14, 1 << 15, 1 << 16};
  static const int64_t batches[] = {1, 4, 16, 32, 64};
  for (int64_t n : sizes) {
    for (int64_t batch : batches) {
      bench->Args({n, batch});
    }
  }
}

static int64_t make_seeded_coeff(std::mt19937_64& rng) {
  return (int64_t)(rng() % 33) - 16;
}

static void init_fft64_input(std::vector<int64_t>& host_input, uint64_t n, uint64_t batch) {
  host_input.resize((size_t)(n * batch));
  std::mt19937_64 rng(0xFF17BEEFULL);
  for (uint64_t p = 0; p < batch; p++) {
    for (uint64_t i = 0; i < n; i++) {
      host_input[p * n + i] = make_seeded_coeff(rng);
    }
  }
}

static bool prepare_fft64_bench(MODULE*& module, double*& d_real, void*& d_freq, uint64_t n, uint64_t batch,
                                benchmark::State& state) {
  module = new_module_info(n, FFT64);
  if (!module || !module->mod.fft64.p_gpu) {
    state.SkipWithError("GPU precomputation is not available for this FFT64 module");
    return false;
  }

  std::vector<int64_t> host_input_i64;
  init_fft64_input(host_input_i64, n, batch);
  std::vector<double> host_input((size_t)(n * batch));
  for (size_t i = 0; i < host_input.size(); i++) {
    host_input[i] = (double)host_input_i64[i];
  }

  const size_t real_count = (size_t)(n * batch);
  const size_t freq_count = (size_t)((n >> 1) * batch);
  if (cudaMalloc((void**)&d_real, real_count * sizeof(double)) != cudaSuccess) {
    state.SkipWithError("cudaMalloc for real buffer failed");
    return false;
  }
  if (cudaMalloc((void**)&d_freq, freq_count * sizeof(double2)) != cudaSuccess) {
    state.SkipWithError("cudaMalloc for freq buffer failed");
    cudaFree(d_real);
    d_real = nullptr;
    return false;
  }
  if (cudaMemcpy(d_real, host_input.data(), real_count * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
    state.SkipWithError("cudaMemcpy H2D for real buffer failed");
    cudaFree(d_freq);
    cudaFree(d_real);
    d_freq = nullptr;
    d_real = nullptr;
    return false;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    state.SkipWithError("cudaDeviceSynchronize failed during FFT64 bench setup");
    cudaFree(d_freq);
    cudaFree(d_real);
    d_freq = nullptr;
    d_real = nullptr;
    return false;
  }
  return true;
}

}  // namespace

static void benchmark_gpu_ffnt_dft(benchmark::State& state) {
  const uint64_t n = (uint64_t)state.range(0);
  const uint64_t batch = (uint64_t)state.range(1);

  MODULE* module = nullptr;
  double* d_real = nullptr;
  void* d_freq = nullptr;
  if (!prepare_fft64_bench(module, d_real, d_freq, n, batch, state)) {
    if (module) {
      delete_module_info(module);
    }
    return;
  }

  for (auto _ : state) {
    if (!fft64_vec_gpu_dft_raw(module, d_freq, d_real, batch)) {
      state.SkipWithError("fft64_vec_gpu_dft_raw failed");
      break;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
      state.SkipWithError("cudaDeviceSynchronize after FFNT failed");
      break;
    }

    state.PauseTiming();
    if (!fft64_vec_gpu_idft_raw(module, d_real, d_freq, batch) || cudaDeviceSynchronize() != cudaSuccess) {
      state.SkipWithError("restore (IFFNT) failed");
      state.ResumeTiming();
      break;
    }
    state.ResumeTiming();
  }

  state.SetItemsProcessed(state.iterations() * (int64_t)(n * batch));
  cudaFree(d_freq);
  cudaFree(d_real);
  delete_module_info(module);
}

static void benchmark_gpu_ffnt_idft(benchmark::State& state) {
  const uint64_t n = (uint64_t)state.range(0);
  const uint64_t batch = (uint64_t)state.range(1);

  MODULE* module = nullptr;
  double* d_real = nullptr;
  void* d_freq = nullptr;
  if (!prepare_fft64_bench(module, d_real, d_freq, n, batch, state)) {
    if (module) {
      delete_module_info(module);
    }
    return;
  }

  if (!fft64_vec_gpu_dft_raw(module, d_freq, d_real, batch) || cudaDeviceSynchronize() != cudaSuccess) {
    state.SkipWithError("initial FFNT preparation failed");
    cudaFree(d_freq);
    cudaFree(d_real);
    delete_module_info(module);
    return;
  }

  for (auto _ : state) {
    if (!fft64_vec_gpu_idft_raw(module, d_real, d_freq, batch)) {
      state.SkipWithError("fft64_vec_gpu_idft_raw failed");
      break;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
      state.SkipWithError("cudaDeviceSynchronize after IFFNT failed");
      break;
    }

    state.PauseTiming();
    if (!fft64_vec_gpu_dft_raw(module, d_freq, d_real, batch) || cudaDeviceSynchronize() != cudaSuccess) {
      state.SkipWithError("restore (FFNT) failed");
      state.ResumeTiming();
      break;
    }
    state.ResumeTiming();
  }

  state.SetItemsProcessed(state.iterations() * (int64_t)(n * batch));
  cudaFree(d_freq);
  cudaFree(d_real);
  delete_module_info(module);
}

BENCHMARK(benchmark_gpu_ffnt_dft)->Name("vec_gpu_ffnt_dft")->Apply(apply_gpu_ffnt_args);
BENCHMARK(benchmark_gpu_ffnt_idft)->Name("vec_gpu_ffnt_idft")->Apply(apply_gpu_ffnt_args);

BENCHMARK_MAIN();
