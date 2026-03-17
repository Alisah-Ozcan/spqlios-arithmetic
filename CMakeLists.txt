cmake_minimum_required(VERSION 3.8)
project(spqlios)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SPQLIOS_USE_GPU "Enable CUDA GPU backends (GPU-NTT/GPU-FFT)" OFF)
set(SPQLIOS_CUDA_ARCH "75" CACHE STRING "CUDA arch list (e.g. 75, 80, 86, 75;86, or native)")
set(SPQLIOS_GPU_NTT_SOURCE "" CACHE PATH "Local path to GPU-NTT source (optional)")
set(SPQLIOS_GPU_NTT_TARGET "GPUNTT::ntt" CACHE STRING "GPU-NTT CMake target name")
set(SPQLIOS_GPU_FFT_SOURCE "" CACHE PATH "Local path to GPU-FFT source (optional)")
set(SPQLIOS_GPU_FFT_TARGET "GPUFFT::fft" CACHE STRING "GPU-FFT CMake target name")

# read the current version from the manifest file
file(READ "manifest.yaml" manifest)
string(REGEX MATCH "version: +(([0-9]+)\\.([0-9]+)\\.([0-9]+))" SPQLIOS_VERSION_BLAH ${manifest})
#message(STATUS "Version: ${SPQLIOS_VERSION_BLAH}")
set(SPQLIOS_VERSION ${CMAKE_MATCH_1})
set(SPQLIOS_VERSION_MAJOR ${CMAKE_MATCH_2})
set(SPQLIOS_VERSION_MINOR ${CMAKE_MATCH_3})
set(SPQLIOS_VERSION_PATCH ${CMAKE_MATCH_4})
message(STATUS "Compiling spqlios-fft version: ${SPQLIOS_VERSION_MAJOR}.${SPQLIOS_VERSION_MINOR}.${SPQLIOS_VERSION_PATCH}")

#set(ENABLE_SPQLIOS_F128 ON CACHE BOOL "Enable float128 via libquadmath")
set(WARNING_PARANOID ON CACHE BOOL "Treat all warnings as errors")
set(ENABLE_TESTING ON CACHE BOOL "Compiles unittests and integration tests")
set(DEVMODE_INSTALL OFF CACHE BOOL "Install private headers and testlib (mainly for CI)")

if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type: Release or Debug" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if (WARNING_PARANOID)
	add_compile_options(
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Werror>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-unused-command-line-argument>
    )
endif()

message(STATUS "CMAKE_HOST_SYSTEM_NAME: ${CMAKE_HOST_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")

if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    set(X86 ON)
    set(AARCH64 OFF)
else ()
    set(X86 OFF)
    # set(ENABLE_SPQLIOS_F128 OFF) # float128 are only supported for x86 targets
endif ()
if (CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)|(arm64)")
    set(AARCH64 ON)
endif ()

if (CMAKE_SYSTEM_NAME MATCHES "(Windows)|(MSYS)")
    set(WIN32 ON)
endif ()
if (WIN32)
    #overrides for win32
    set(X86 OFF)
    set(AARCH64 OFF)
    set(X86_WIN32 ON)
else()
    set(X86_WIN32 OFF)
    set(WIN32 OFF)
endif (WIN32)

message(STATUS "--> WIN32: ${WIN32}")
message(STATUS "--> X86_WIN32: ${X86_WIN32}")
message(STATUS "--> X86_LINUX: ${X86}")
message(STATUS "--> AARCH64: ${AARCH64}")

if (SPQLIOS_USE_GPU)
    enable_language(CUDA)
    # Prefer shared CUDA runtime to avoid static cudart registration issues
    # when CUDA kernels are linked through shared libraries.
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
    string(TOLOWER "${SPQLIOS_CUDA_ARCH}" _spqlios_cuda_arch_cfg)
    if (_spqlios_cuda_arch_cfg STREQUAL "native")
        # NOTE: In headless/containerized builds, NVCC native may resolve to a single
        # architecture that does not match runtime GPU. Build a small fat binary set.
        set(_spqlios_cuda_arches "75;80;86;89")
        message(WARNING "SPQLIOS_CUDA_ARCH=native detected. Using fat CUDA arch set: ${_spqlios_cuda_arches}")
    else()
        string(REPLACE "," ";" _spqlios_cuda_arches "${SPQLIOS_CUDA_ARCH}")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "${_spqlios_cuda_arches}" CACHE STRING "CUDA architectures" FORCE)
    message(STATUS "SPQLIOS CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    if (POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
    endif()
    include(FetchContent)
    function(spqlios_add_gpu_dependency dep_name dep_source_var dep_target_var dep_fetched_dir_var dep_git_repository dep_git_tag)
        if (${dep_source_var})
            FetchContent_Declare(
                    ${dep_name}
                    SOURCE_DIR "${${dep_source_var}}"
            )
        else()
            FetchContent_Declare(
                    ${dep_name}
                    GIT_REPOSITORY ${dep_git_repository}
                    GIT_TAG ${dep_git_tag}
            )
        endif()

        FetchContent_GetProperties(${dep_name})
        if (NOT ${dep_name}_POPULATED)
            FetchContent_Populate(${dep_name})
        endif()

        string(CONCAT dep_source_dir_var "${dep_name}" "_SOURCE_DIR")
        set(dep_source_dir "${${dep_source_dir_var}}")
        set(${dep_fetched_dir_var} "${dep_source_dir}" CACHE INTERNAL "${dep_name} source dir" FORCE)

        file(GLOB_RECURSE dep_sources CONFIGURE_DEPENDS
                "${dep_source_dir}/*.cu"
                "${dep_source_dir}/*.c"
                "${dep_source_dir}/*.cc"
                "${dep_source_dir}/*.cpp"
                "${dep_source_dir}/*.cxx"
        )
        foreach (pat ".*[/\\\\]test[/\\\\].*"
                     ".*[/\\\\]tests[/\\\\].*"
                     ".*[/\\\\]example[/\\\\].*"
                     ".*[/\\\\]examples[/\\\\].*"
                     ".*[/\\\\]benchmark[/\\\\].*"
                     ".*[/\\\\]benchmarks[/\\\\].*"
                     ".*[/\\\\]docs[/\\\\].*")
            list(FILTER dep_sources EXCLUDE REGEX "${pat}")
        endforeach()

        if (dep_sources STREQUAL "")
            message(FATAL_ERROR "${dep_name}: no buildable sources found in ${dep_source_dir}")
        endif()

        set(dep_target "${dep_name}_external")
        add_library(${dep_target} STATIC ${dep_sources})
        set_property(TARGET ${dep_target} PROPERTY POSITION_INDEPENDENT_CODE ON)
        set_property(TARGET ${dep_target} PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
        set_property(TARGET ${dep_target} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        set_property(TARGET ${dep_target} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
        target_include_directories(${dep_target} PUBLIC "${dep_source_dir}")
        if (EXISTS "${dep_source_dir}/include")
            target_include_directories(${dep_target} PUBLIC "${dep_source_dir}/include")
        endif()
        if (EXISTS "${dep_source_dir}/src/include")
            target_include_directories(${dep_target} PUBLIC "${dep_source_dir}/src/include")
        endif()
        if (EXISTS "${dep_source_dir}/src")
            target_include_directories(${dep_target} PUBLIC "${dep_source_dir}/src")
        endif()
        set(${dep_target_var} "${dep_target}" CACHE INTERNAL "${dep_name} target" FORCE)
    endfunction()

    spqlios_add_gpu_dependency(
            gpu_ntt
            SPQLIOS_GPU_NTT_SOURCE
            SPQLIOS_GPU_NTT_TARGET
            SPQLIOS_GPU_NTT_FETCHED_DIR
            https://github.com/Alisah-Ozcan/GPU-NTT.git
            main
    )
    spqlios_add_gpu_dependency(
            gpu_fft
            SPQLIOS_GPU_FFT_SOURCE
            SPQLIOS_GPU_FFT_TARGET
            SPQLIOS_GPU_FFT_FETCHED_DIR
            https://github.com/Alisah-Ozcan/GPU-FFT.git
            main
    )

    set(SPQLIOS_GPU_EXTERNAL_TARGETS "")
    if (TARGET ${SPQLIOS_GPU_NTT_TARGET})
        list(APPEND SPQLIOS_GPU_EXTERNAL_TARGETS ${SPQLIOS_GPU_NTT_TARGET})
        set(SPQLIOS_HAVE_GPUNTT ON)
    endif()
    if (TARGET ${SPQLIOS_GPU_FFT_TARGET})
        list(APPEND SPQLIOS_GPU_EXTERNAL_TARGETS ${SPQLIOS_GPU_FFT_TARGET})
        set(SPQLIOS_HAVE_GPUFFT ON)
    endif()
    set(SPQLIOS_GPU_EXTERNAL_TARGETS "${SPQLIOS_GPU_EXTERNAL_TARGETS}" CACHE INTERNAL "GPU dependency targets" FORCE)
endif()

# compiles the main library in spqlios
add_subdirectory(spqlios)

# compiles and activates unittests and itests
if (${ENABLE_TESTING})
    enable_testing()
    add_subdirectory(test)
endif()

# build local examples if present
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/example/CMakeLists.txt")
    add_subdirectory(example)
endif()
