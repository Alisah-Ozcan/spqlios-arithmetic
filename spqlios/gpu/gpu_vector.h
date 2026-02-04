#ifndef SPQLIOS_GPU_VEC_H
#define SPQLIOS_GPU_VEC_H

#include <stdint.h>

#include "../commons.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <stdexcept>
#include <string>

static inline void spqlios_cuda_check(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error: ") + msg + ": " + cudaGetErrorString(err));
  }
}

// C++ helper for GPU buffers (not part of the C API)
template <typename T>
class VEC_GPU {
 public:
  VEC_GPU() : d_ptr_(nullptr), size_(0) {}

  explicit VEC_GPU(size_t size) : VEC_GPU() { allocate(size); }

  VEC_GPU(const T* host_ptr, size_t size) : VEC_GPU() { copy_from_host(host_ptr, size); }

  VEC_GPU(const VEC_GPU& other) : VEC_GPU() { copy_from(other); }
  VEC_GPU& operator=(const VEC_GPU& other) {
    if (this != &other) {
      reset();
      copy_from(other);
    }
    return *this;
  }

  VEC_GPU(VEC_GPU&& other) noexcept { move_from(other); }
  VEC_GPU& operator=(VEC_GPU&& other) noexcept {
    if (this != &other) {
      reset();
      move_from(other);
    }
    return *this;
  }

  ~VEC_GPU() { reset(); }

  void reset() {
    if (d_ptr_) {
      (void)cudaFree(d_ptr_);
    }
    d_ptr_ = nullptr;
    size_ = 0;
  }

  void allocate(size_t size) {
    reset();
    if (size == 0) {
      return;
    }
    spqlios_cuda_check(cudaMalloc((void**)&d_ptr_, size * sizeof(T)), "cudaMalloc");
    size_ = size;
  }

  void copy_from_host(const T* host_ptr, size_t size) {
    if (size == 0) {
      reset();
      return;
    }
    if (!host_ptr) {
      fprintf(stderr, "VEC_GPU::copy_from_host: null host pointer\n");
      abort();
    }
    if (size_ != size) {
      allocate(size);
    }
    spqlios_cuda_check(cudaMemcpy(d_ptr_, host_ptr, size * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
  }

  void copy_to_host(T* host_ptr, size_t size) const {
    if (size == 0) {
      return;
    }
    if (!host_ptr) {
      fprintf(stderr, "VEC_GPU::copy_to_host: null host pointer\n");
      abort();
    }
    if (size > size_) {
      fprintf(stderr, "VEC_GPU::copy_to_host: size exceeds device buffer\n");
      abort();
    }
    spqlios_cuda_check(cudaMemcpy(host_ptr, d_ptr_, size * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
  }

  T* data() const { return d_ptr_; }
  size_t size() const { return size_; }
  explicit operator bool() const { return d_ptr_ != nullptr; }

  operator T*() { return d_ptr_; }
  operator const T*() const { return d_ptr_; }

 private:
  void copy_from(const VEC_GPU& other) {
    if (!other.d_ptr_ || other.size_ == 0) {
      return;
    }
    allocate(other.size_);
    spqlios_cuda_check(cudaMemcpy(d_ptr_, other.d_ptr_, other.size_ * sizeof(T), cudaMemcpyDeviceToDevice),
                       "cudaMemcpy D2D");
  }
  void move_from(VEC_GPU& other) {
    d_ptr_ = other.d_ptr_;
    size_ = other.size_;
    other.d_ptr_ = nullptr;
    other.size_ = 0;
  }

  T* d_ptr_;
  size_t size_;
};
#endif

#endif  // SPQLIOS_GPU_VEC_H
