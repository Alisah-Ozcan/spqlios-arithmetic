#include "ntt_parameters.cuh"

static int log2_u64_impl(uint64_t n) {
  int logn = 0;
  uint64_t v = 1;
  while (v < n) {
    v <<= 1;
    logn++;
  }
  if (v != n) {
    return -1;
  }
  return logn;
}

int NTTParameterModule::log2_u64(uint64_t n) { return log2_u64_impl(n); }

NTTParameterModule::NTTParameterModule(uint64_t n, gpuntt::ReductionPolynomial poly)
    : logn_(log2_u64(n)), n_(n), poly_(poly) {
  if (logn_ < 1) {
    throw std::runtime_error("NTTParameterModule: n must be power of two");
  }
  if (logn_ > 25) {
    throw std::runtime_error("NTTParameterModule: logn must be <= 25 for Data32");
  }

  gpuntt::NTTParameters<Data32> params(logn_, poly_);
  modulus_ = params.modulus;
  n_inv_ = params.n_inv;

  std::vector<Root<Data32>> forward = params.gpu_root_of_unity_table_generator(params.forward_root_of_unity_table);
  std::vector<Root<Data32>> inverse = params.gpu_root_of_unity_table_generator(params.inverse_root_of_unity_table);

  forward_table_.copy_from_host(forward.data(), forward.size());
  inverse_table_.copy_from_host(inverse.data(), inverse.size());
}
