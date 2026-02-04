#ifndef SPQLIOS_GPU_NTT_PARAMETERS_H
#define SPQLIOS_GPU_NTT_PARAMETERS_H

#include <stdint.h>

#include <stdexcept>
#include <vector>

#include "gpu_vector.h"
#include "gpuntt/common/modular_arith.cuh"
#include "gpuntt/common/nttparameters.cuh"

class NTTParameterModule {
 public:
  explicit NTTParameterModule(uint64_t n, gpuntt::ReductionPolynomial poly = gpuntt::ReductionPolynomial::X_N_plus);

  uint64_t n() const { return n_; }
  int logn() const { return logn_; }
  gpuntt::ReductionPolynomial poly() const { return poly_; }

  Modulus<Data32> modulus() const { return modulus_; }
  Ninverse<Data32> n_inv() const { return n_inv_; }

  Root<Data32>* forward_table() const { return forward_table_.data(); }
  Root<Data32>* inverse_table() const { return inverse_table_.data(); }

 private:
  static int log2_u64(uint64_t n);

  int logn_;
  uint64_t n_;
  gpuntt::ReductionPolynomial poly_;
  Modulus<Data32> modulus_;
  Ninverse<Data32> n_inv_;
  VEC_GPU<Root<Data32>> forward_table_;
  VEC_GPU<Root<Data32>> inverse_table_;
};

#endif  // SPQLIOS_GPU_NTT_PARAMETERS_H
