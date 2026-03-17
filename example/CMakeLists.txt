add_executable(ntt_mul_batch_demo ntt_mul_batch_demo.c)
target_include_directories(ntt_mul_batch_demo PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(ntt_mul_batch_demo libspqlios)

add_executable(vec_znx_dft_dispatch_demo vec_znx_dft_dispatch_demo.c)
target_include_directories(vec_znx_dft_dispatch_demo PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(vec_znx_dft_dispatch_demo libspqlios)

add_executable(vec_znx_idft_dispatch_demo vec_znx_idft_dispatch_demo.c)
target_include_directories(vec_znx_idft_dispatch_demo PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(vec_znx_idft_dispatch_demo libspqlios)

if (SPQLIOS_USE_GPU)
    add_executable(fft64_ffnt_dispatch_demo fft64_ffnt_dispatch_demo.c)
    target_include_directories(fft64_ffnt_dispatch_demo PRIVATE ${PROJECT_SOURCE_DIR})
    target_link_libraries(fft64_ffnt_dispatch_demo libspqlios)

    add_executable(fft64_ffnt_idft_dispatch_demo fft64_ffnt_idft_dispatch_demo.c)
    target_include_directories(fft64_ffnt_idft_dispatch_demo PRIVATE ${PROJECT_SOURCE_DIR})
    target_link_libraries(fft64_ffnt_idft_dispatch_demo libspqlios)

    add_executable(fft64_ffnt_mul_demo fft64_ffnt_mul_demo.c)
    target_include_directories(fft64_ffnt_mul_demo PRIVATE ${PROJECT_SOURCE_DIR})
    target_link_libraries(fft64_ffnt_mul_demo libspqlios)

    add_executable(ntt_gpu_demo ntt_gpu_demo.cu)
    target_include_directories(ntt_gpu_demo PRIVATE ${PROJECT_SOURCE_DIR})
    target_link_libraries(ntt_gpu_demo libspqlios-gpu)
endif()
