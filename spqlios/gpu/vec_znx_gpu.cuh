#ifndef SPQLIOS_VEC_ZNX_GPU_H
#define SPQLIOS_VEC_ZNX_GPU_H

#include <stdint.h>

#include "gpu_vector.h"
#include "ntt_parameters.cuh"

#ifdef __cplusplus

// GPU NTT helpers using VEC_GPU buffers (contiguous layout)
void vec_znx_dft_gpu(VEC_GPU<uint32_t>& out, const uint32_t* host_in, const NTTParameterModule& params, uint64_t batch);
void vec_znx_dft_gpu(VEC_GPU<uint32_t>& out, const VEC_GPU<uint32_t>& in, const NTTParameterModule& params,
                     uint64_t batch);
void vec_dft_hadamard_gpu(VEC_GPU<uint32_t>& out, const VEC_GPU<uint32_t>& a, const VEC_GPU<uint32_t>& b,
                          const NTTParameterModule& params, uint64_t batch);
void vec_znx_idft_gpu(uint32_t* host_out, VEC_GPU<uint32_t>& in, const NTTParameterModule& params, uint64_t batch);

#endif  // __cplusplus

#endif  // SPQLIOS_VEC_ZNX_GPU_H
