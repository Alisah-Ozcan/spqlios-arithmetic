add_executable(ntt_mul_batch_demo ntt_mul_batch_demo.c)
target_include_directories(ntt_mul_batch_demo PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(ntt_mul_batch_demo libspqlios)

if (SPQLIOS_ENABLE_GPU)
    add_executable(ntt_gpu_demo ntt_gpu_demo.cu)
    target_include_directories(ntt_gpu_demo PRIVATE ${PROJECT_SOURCE_DIR})
    target_link_libraries(ntt_gpu_demo libspqlios-gpu)
endif()
