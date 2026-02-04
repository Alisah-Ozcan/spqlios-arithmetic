cmake_minimum_required(VERSION 3.8)
project(spqlios)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SPQLIOS_ENABLE_GPU "Enable CUDA GPU backend (GPU-NTT)" OFF)
set(SPQLIOS_CUDA_ARCH "75" CACHE STRING "CUDA arch (e.g. 75, 80, 86)")
set(SPQLIOS_GPU_NTT_SOURCE "" CACHE PATH "Local path to GPU-NTT source (optional)")
set(SPQLIOS_GPU_NTT_TARGET "GPUNTT::ntt" CACHE STRING "GPU-NTT CMake target name")

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

if (SPQLIOS_ENABLE_GPU)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES ${SPQLIOS_CUDA_ARCH} CACHE STRING "CUDA architectures" FORCE)
    include(FetchContent)
    if (SPQLIOS_GPU_NTT_SOURCE)
        FetchContent_Declare(
                gpu_ntt
                SOURCE_DIR ${SPQLIOS_GPU_NTT_SOURCE}
        )
    else()
        FetchContent_Declare(
                gpu_ntt
                GIT_REPOSITORY https://github.com/Alisah-Ozcan/GPU-NTT.git
                GIT_TAG main
        )
    endif()
    FetchContent_Populate(gpu_ntt)
    set(SPQLIOS_GPU_NTT_FETCHED_DIR ${gpu_ntt_SOURCE_DIR} CACHE INTERNAL "GPU-NTT source dir")
    set(SPQLIOS_HAVE_GPUNTT ON)

    # Build a local target directly from fetched sources (no external target name required)
    file(GLOB_RECURSE GPU_NTT_SOURCES CONFIGURE_DEPENDS
            "${gpu_ntt_SOURCE_DIR}/*.cu"
            "${gpu_ntt_SOURCE_DIR}/*.c"
            "${gpu_ntt_SOURCE_DIR}/*.cpp"
    )
    foreach (pat ".*[/\\\\]test[/\\\\].*"
                 ".*[/\\\\]tests[/\\\\].*"
                 ".*[/\\\\]example[/\\\\].*"
                 ".*[/\\\\]examples[/\\\\].*"
                 ".*[/\\\\]benchmark[/\\\\].*"
                 ".*[/\\\\]benchmarks[/\\\\].*"
                 ".*[/\\\\]docs[/\\\\].*")
        list(FILTER GPU_NTT_SOURCES EXCLUDE REGEX "${pat}")
    endforeach()
    if (GPU_NTT_SOURCES STREQUAL "")
        message(FATAL_ERROR "GPU-NTT sources not found in ${gpu_ntt_SOURCE_DIR}")
    endif()
    add_library(gpu_ntt_external STATIC ${GPU_NTT_SOURCES})
    target_include_directories(gpu_ntt_external PUBLIC ${gpu_ntt_SOURCE_DIR})
    if (EXISTS "${gpu_ntt_SOURCE_DIR}/include")
        target_include_directories(gpu_ntt_external PUBLIC ${gpu_ntt_SOURCE_DIR}/include)
    endif()
    if (EXISTS "${gpu_ntt_SOURCE_DIR}/src/include")
        target_include_directories(gpu_ntt_external PUBLIC ${gpu_ntt_SOURCE_DIR}/src/include)
    endif()
    if (EXISTS "${gpu_ntt_SOURCE_DIR}/src")
        target_include_directories(gpu_ntt_external PUBLIC ${gpu_ntt_SOURCE_DIR}/src)
    endif()
    set_property(TARGET gpu_ntt_external PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set(SPQLIOS_GPU_NTT_TARGET gpu_ntt_external CACHE INTERNAL "GPU-NTT target" FORCE)
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
