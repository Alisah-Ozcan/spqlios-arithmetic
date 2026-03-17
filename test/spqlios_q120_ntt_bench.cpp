#include <benchmark/benchmark.h>

#include <cstdint>
#include <vector>

#include "spqlios/q120/q120_ntt.h"

static void apply_ntt_args(benchmark::internal::Benchmark* bench) {
  static const int64_t sizes[] = {1 << 11, 1 << 12, 1 << 13, 1 << 14, 1 << 15, 1 << 16};
  static const int64_t batches[] = {1, 4, 16, 32, 64};
  for (int64_t n : sizes) {
    for (int64_t batch : batches) {
      bench->Args({n, batch});
    }
  }
}

template <typeof(q120_ntt_bb_avx2) f>
void benchmark_ntt(benchmark::State& state) {
  const uint64_t n = state.range(0);
  const uint64_t batch = state.range(1);
  q120_ntt_precomp* precomp = q120_new_ntt_bb_precomp(n);

  std::vector<uint64_t> px(n * 4 * batch);
  for (uint64_t i = 0; i < 4 * n * batch; i++) {
    px[i] = (rand() << 31) + rand();
  }
  for (auto _ : state) {
    for (uint64_t b = 0; b < batch; b++) {
      f(precomp, (q120b*)(px.data() + b * n * 4));
    }
  }
  state.SetItemsProcessed(state.iterations() * (int64_t)(n * batch));
  q120_del_ntt_bb_precomp(precomp);
}

template <typeof(q120_intt_bb_avx2) f>
void benchmark_intt(benchmark::State& state) {
  const uint64_t n = state.range(0);
  const uint64_t batch = state.range(1);
  q120_ntt_precomp* precomp = q120_new_intt_bb_precomp(n);

  std::vector<uint64_t> px(n * 4 * batch);
  for (uint64_t i = 0; i < 4 * n * batch; i++) {
    px[i] = (rand() << 31) + rand();
  }
  for (auto _ : state) {
    for (uint64_t b = 0; b < batch; b++) {
      f(precomp, (q120b*)(px.data() + b * n * 4));
    }
  }
  state.SetItemsProcessed(state.iterations() * (int64_t)(n * batch));
  q120_del_intt_bb_precomp(precomp);
}

BENCHMARK(benchmark_ntt<q120_ntt_bb_avx2>)->Name("q120_ntt_bb_avx2")->Apply(apply_ntt_args);
BENCHMARK(benchmark_intt<q120_intt_bb_avx2>)->Name("q120_intt_bb_avx2")->Apply(apply_ntt_args);

BENCHMARK_MAIN();
