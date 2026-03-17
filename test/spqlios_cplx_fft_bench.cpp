#include <benchmark/benchmark.h>
#include <stdint.h>

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "../spqlios/cplx/cplx_fft_internal.h"
#include "spqlios/reim/reim_fft.h"

using namespace std;

void init_random_values(uint64_t n, double* v) {
  for (uint64_t i = 0; i < n; ++i) v[i] = rand() - (RAND_MAX >> 1);
}

static void apply_fft_args(benchmark::internal::Benchmark* bench) {
  static const int64_t sizes[] = {2048, 4096, 8192, 16384, 32768, 65536};
  static const int64_t batches[] = {1, 4, 16, 32, 64};
  for (int64_t nn : sizes) {
    for (int64_t batch : batches) {
      bench->Args({nn, batch});
    }
  }
}

void benchmark_cplx_fft(benchmark::State& state) {
  const int32_t nn = state.range(0);
  const uint32_t batch = (uint32_t)state.range(1);
  CPLX_FFT_PRECOMP* a = new_cplx_fft_precomp(nn / 2, batch);
  for (uint32_t i = 0; i < batch; i++) {
    double* c = (double*)cplx_fft_precomp_get_buffer(a, i);
    init_random_values(nn, c);
  }
  for (auto _ : state) {
    for (uint32_t i = 0; i < batch; i++) {
      double* c = (double*)cplx_fft_precomp_get_buffer(a, i);
      cplx_fft(a, c);
    }
  }
  state.SetItemsProcessed(state.iterations() * (int64_t)(nn * batch));
  delete_cplx_fft_precomp(a);
}

void benchmark_cplx_ifft(benchmark::State& state) {
  const int32_t nn = state.range(0);
  const uint32_t batch = (uint32_t)state.range(1);
  CPLX_IFFT_PRECOMP* a = new_cplx_ifft_precomp(nn / 2, batch);
  for (uint32_t i = 0; i < batch; i++) {
    double* c = (double*)cplx_ifft_precomp_get_buffer(a, i);
    init_random_values(nn, c);
  }
  for (auto _ : state) {
    for (uint32_t i = 0; i < batch; i++) {
      double* c = (double*)cplx_ifft_precomp_get_buffer(a, i);
      cplx_ifft(a, c);
    }
  }
  state.SetItemsProcessed(state.iterations() * (int64_t)(nn * batch));
  delete_cplx_ifft_precomp(a);
}

void benchmark_reim_fft(benchmark::State& state) {
  const int32_t nn = state.range(0);
  const uint32_t m = nn / 2;
  const uint32_t batch = (uint32_t)state.range(1);
  REIM_FFT_PRECOMP* a = new_reim_fft_precomp(m, batch);
  for (uint32_t i = 0; i < batch; i++) {
    double* c = reim_fft_precomp_get_buffer(a, i);
    init_random_values(nn, c);
  }
  for (auto _ : state) {
    for (uint32_t i = 0; i < batch; i++) {
      double* c = reim_fft_precomp_get_buffer(a, i);
      reim_fft(a, c);
    }
  }
  state.SetItemsProcessed(state.iterations() * (int64_t)(nn * batch));
  delete_reim_fft_precomp(a);
}

#ifdef __aarch64__
EXPORT REIM_FFT_PRECOMP* new_reim_fft_precomp_neon(uint32_t m, uint32_t num_buffers);
EXPORT void reim_fft_neon(const REIM_FFT_PRECOMP* precomp, double* d);

void benchmark_reim_fft_neon(benchmark::State& state) {
  const int32_t nn = state.range(0);
  const uint32_t m = nn / 2;
  const uint32_t batch = (uint32_t)state.range(1);
  REIM_FFT_PRECOMP* a = new_reim_fft_precomp_neon(m, batch);
  for (uint32_t i = 0; i < batch; i++) {
    double* c = reim_fft_precomp_get_buffer(a, i);
    init_random_values(nn, c);
  }
  for (auto _ : state) {
    for (uint32_t i = 0; i < batch; i++) {
      double* c = reim_fft_precomp_get_buffer(a, i);
      reim_fft_neon(a, c);
    }
  }
  state.SetItemsProcessed(state.iterations() * (int64_t)(nn * batch));
  delete_reim_fft_precomp(a);
}
#endif

void benchmark_reim_ifft(benchmark::State& state) {
  const int32_t nn = state.range(0);
  const uint32_t m = nn / 2;
  const uint32_t batch = (uint32_t)state.range(1);
  REIM_IFFT_PRECOMP* a = new_reim_ifft_precomp(m, batch);
  for (uint32_t i = 0; i < batch; i++) {
    double* c = reim_ifft_precomp_get_buffer(a, i);
    init_random_values(nn, c);
  }
  for (auto _ : state) {
    for (uint32_t i = 0; i < batch; i++) {
      double* c = reim_ifft_precomp_get_buffer(a, i);
      reim_ifft(a, c);
    }
  }
  state.SetItemsProcessed(state.iterations() * (int64_t)(nn * batch));
  delete_reim_ifft_precomp(a);
}

int main(int argc, char** argv) {
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  std::cout << "Dimensions n in the benchmark below are in \"real FFT\" modulo X^n+1" << std::endl;
  std::cout << "The complex dimension m (modulo X^m-i) is half of it" << std::endl;
  BENCHMARK(benchmark_cplx_fft)->Apply(apply_fft_args);
  BENCHMARK(benchmark_cplx_ifft)->Apply(apply_fft_args);
  BENCHMARK(benchmark_reim_fft)->Apply(apply_fft_args);
#ifdef __aarch64__
  BENCHMARK(benchmark_reim_fft_neon)->Apply(apply_fft_args);
#endif
  BENCHMARK(benchmark_reim_ifft)->Apply(apply_fft_args);
  // if (CPU_SUPPORTS("avx512f")) {
  //  BENCHMARK(bench_cplx_fftvec_twiddle_avx512)->ARGS;
  //  BENCHMARK(bench_cplx_fftvec_bitwiddle_avx512)->ARGS;
  //}
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}
