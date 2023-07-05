cmake_minimum_required(VERSION 3.12)

project("rwkv.cpp" C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#
# Option list
#

# General
option(RWKV_BUILD_SHARED_LIBRARY   "rwkv: build as a shared library"                      ON)
option(RWKV_STATIC                 "rwkv: static link libraries"                          OFF)
option(RWKV_NATIVE                 "rwkv: enable -march=native flag"                      OFF)
option(RWKV_LTO                    "rwkv: enable link time optimization"                  OFF)

# Debug
option(RWKV_ALL_WARNINGS           "rwkv: enable all compiler warnings"                   ON)
option(RWKV_GPROF                  "rwkv: enable gprof"                                   OFF)

# Sanitizers
option(RWKV_SANITIZE_THREAD        "rwkv: enable thread sanitizer"                        OFF)
option(RWKV_SANITIZE_ADDRESS       "rwkv: enable address sanitizer"                       OFF)
option(RWKV_SANITIZE_UNDEFINED     "rwkv: enable undefined sanitizer"                     OFF)

# Instruction set specific
option(RWKV_AVX                    "rwkv: enable AVX"                                     ON)
option(RWKV_AVX2                   "rwkv: enable AVX2"                                    ON)
option(RWKV_AVX512                 "rwkv: enable AVX512"                                  OFF)
option(RWKV_FMA                    "rwkv: enable FMA"                                     ON)

# 3rd party libs
option(RWKV_ACCELERATE             "rwkv: enable Accelerate framework"                    ON)
option(RWKV_OPENBLAS               "rwkv: use OpenBLAS"                                   OFF)
option(RWKV_CUBLAS                 "rwkv: use cuBLAS"                                     OFF)
option(RWKV_CLBLAST                "rwkv: use CLBlast"                                    OFF)

# Build only shared library without building tests and extras
option(RWKV_STANDALONE             "rwkv: build only RWKV library"                        OFF)

#
# Compile flags
#

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if (NOT MSVC)
    if (RWKV_SANITIZE_THREAD)
        add_compile_options(-fsanitize=thread)
        link_libraries(-fsanitize=thread)
    endif()

    if (RWKV_SANITIZE_ADDRESS)
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        link_libraries(-fsanitize=address)
    endif()

    if (RWKV_SANITIZE_UNDEFINED)
        add_compile_options(-fsanitize=undefined)
        link_libraries(-fsanitize=undefined)
    endif()
endif()

if (APPLE AND RWKV_ACCELERATE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if (ACCELERATE_FRAMEWORK)
        message(STATUS "Accelerate framework found")

        add_compile_definitions(GGML_USE_ACCELERATE)
        set(RWKV_EXTRA_LIBS ${RWKV_EXTRA_LIBS} ${ACCELERATE_FRAMEWORK})
    else()
        message(WARNING "Accelerate framework not found")
    endif()
endif()

if (RWKV_OPENBLAS)
    if (RWKV_STATIC)
        set(BLA_STATIC ON)
    endif()

    set(BLA_VENDOR OpenBLAS)
    find_package(BLAS)
    if (BLAS_FOUND)
        message(STATUS "OpenBLAS found")

        add_compile_definitions(GGML_USE_OPENBLAS)
        add_link_options(${BLAS_LIBRARIES})
    else()
        message(WARNING "OpenBLAS not found")
    endif()
endif()

if (RWKV_CUBLAS)
    cmake_minimum_required(VERSION 3.17)

    find_package(CUDAToolkit)
    if (CUDAToolkit_FOUND)
        message(STATUS "cuBLAS found")

        enable_language(CUDA)

        set(GGML_CUDA_SOURCES ${CMAKE_SOURCE_DIR}/ggml/src/ggml-cuda.cu ${CMAKE_SOURCE_DIR}/ggml/src/ggml-cuda.h)

        add_compile_definitions(GGML_USE_CUBLAS)

        if (RWKV_STATIC)
            set(RWKV_EXTRA_LIBS ${RWKV_EXTRA_LIBS} CUDA::cudart_static CUDA::cublas_static CUDA::cublasLt_static)
        else()
            set(RWKV_EXTRA_LIBS ${RWKV_EXTRA_LIBS} CUDA::cudart CUDA::cublas CUDA::cublasLt)
        endif()

    else()
        message(WARNING "cuBLAS not found")
    endif()
endif()

if (RWKV_CLBLAST)
    cmake_minimum_required(VERSION 3.17)

    file(GLOB OPENCL_INCLUDE_SEARCH_PATHS ${CMAKE_SOURCE_DIR}/OpenCL-*/)
    file(GLOB CLBLAST_INCLUDE_SEARCH_PATHS ${CMAKE_SOURCE_DIR}/CLBlast-*/)

    set(OPENCL_INCLUDE_SEARCH_PATHS
        /usr/include
        /usr/local/include
        $ENV{OPENCL_HOME}
        $ENV{OPENCL_HOME}/include
        ${OPENCL_INCLUDE_SEARCH_PATHS}
        )

    set(CLBLAST_INCLUDE_SEARCH_PATHS
        /usr/include
        /usr/local/include
        $ENV{CLBLAST_HOME}
        $ENV{CLBLAST_HOME}/include
        ${CLBLAST_INCLUDE_SEARCH_PATHS}
        )

    find_path(OPENCL_INC NAMES opencl.h PATHS ${OPENCL_INCLUDE_SEARCH_PATHS} PATH_SUFFIXES include/CL)
    find_library(OPENCL_LIB NAMES OpenCL PATHS ${OPENCL_INCLUDE_SEARCH_PATHS} PATH_SUFFIXES lib)
    find_path(CLBLAST_INC NAMES clblast.h PATHS ${CLBLAST_INCLUDE_SEARCH_PATHS} PATH_SUFFIXES include)
    find_library(CLBLAST_LIB NAMES clblast PATHS ${CLBLAST_INCLUDE_SEARCH_PATHS} PATH_SUFFIXES lib)

    if (OPENCL_LIB)
        set(OPENCL_INC ${OPENCL_INC}/..) # disgusting
        message(STATUS "OpenCL SDK found: ${OPENCL_INC}")

        if (CLBLAST_LIB)
            message(STATUS "CLBlast found: ${CLBLAST_INC}")
            add_compile_definitions(GGML_USE_CLBLAST)
            set(GGML_OPENCL_SOURCES ${CMAKE_SOURCE_DIR}/ggml/src/ggml-opencl.cpp ${CMAKE_SOURCE_DIR}/ggml/src/ggml-opencl.h)
            set(GGML_OPENCL_DIRS ${GGML_OPENCL_DIRS} ${OPENCL_INC} ${CLBLAST_INC})
            set(RWKV_EXTRA_LIBS ${RWKV_EXTRA_LIBS} ${OPENCL_LIB} ${CLBLAST_LIB})
            link_libraries("-Wl,--copy-dt-needed-entries")
        else()
            message(WARNING "CLBlast not found")
        endif()
    else()
       message(WARNING "OpenCL SDK not found, CLBlast cannot be enabled")
    endif()
endif()

if (RWKV_ALL_WARNINGS)
    if (NOT MSVC)
        set(c_flags
            -Wall
            -Wextra
            -Wpedantic
            -Wcast-qual
            -Wdouble-promotion
            -Wshadow
            -Wstrict-prototypes
            -Wpointer-arith
            -Wno-unused-function
        )
        set(cxx_flags
            -Wall
            -Wextra
            -Wpedantic
            -Wcast-qual
            -Wno-unused-function
            -Wno-multichar
        )
    else()
        set(c_flags
            -W4
        )
        set(cxx_flags
            -W4
        )
    endif()

    add_compile_options(
            "$<$<COMPILE_LANGUAGE:C>:${c_flags}>"
            "$<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>"
    )

endif()

if (RWKV_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if (result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "IPO is not supported: ${output}")
    endif()
endif()

# Architecture specific
# TODO [llama.cpp]: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if (NOT MSVC)
    if (RWKV_STATIC)
        add_link_options(-static)
        if (MINGW)
            add_link_options(-static-libgcc -static-libstdc++)
        endif()
    endif()
    if (RWKV_GPROF)
        add_compile_options(-pg)
    endif()
    if (RWKV_NATIVE)
        add_compile_options(-march=native)
    endif()
endif()

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    message(STATUS "ARM detected")
    if (MSVC)
        # TODO [llama.cpp]: arm msvc?
    else()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
            add_compile_options(-mcpu=native)
        endif()
        # TODO [llama.cpp]: armv6,7,8 version specific flags
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(x86_64|i686|AMD64)$")
    message(STATUS "x86 detected")
    if (MSVC)
        if (RWKV_AVX512)
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX512>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
            # MSVC has no compile-time flags enabling specific
            # AVX512 extensions, neither it defines the
            # macros corresponding to the extensions.
            # Do it manually.
        elseif (RWKV_AVX2)
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX2>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
        elseif (RWKV_AVX)
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX>)
        endif()
    else()
        add_compile_options(-mf16c)
        if (RWKV_FMA)
            add_compile_options(-mfma)
        endif()
        if (RWKV_AVX)
            add_compile_options(-mavx)
        endif()
        if (RWKV_AVX2)
            add_compile_options(-mavx2)
        endif()
        if (RWKV_AVX512)
            add_compile_options(-mavx512f)
            add_compile_options(-mavx512bw)
        endif()
    endif()
else()
    # TODO [llama.cpp]: support PowerPC
    message(STATUS "Unknown architecture")
endif()

#
# Build libraries
#

if (MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

add_library(ggml OBJECT
            ${CMAKE_SOURCE_DIR}/ggml/src/ggml.c
            ${CMAKE_SOURCE_DIR}/ggml/include/ggml/ggml.h
            ${GGML_CUDA_SOURCES}
            ${GGML_OPENCL_SOURCES})

target_include_directories(ggml PUBLIC ${CMAKE_SOURCE_DIR}/ggml/include/ggml ${GGML_OPENCL_DIRS})
target_compile_features(ggml PUBLIC c_std_11) # Don't bump

if (MSVC)
    target_link_libraries(ggml PUBLIC ${RWKV_EXTRA_LIBS} Threads::Threads)
else()
    target_link_libraries(ggml PUBLIC m ${RWKV_EXTRA_LIBS} Threads::Threads)
endif()

if (RWKV_BUILD_SHARED_LIBRARY)
    set_target_properties(ggml PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if (RWKV_BUILD_SHARED_LIBRARY)
    add_library(rwkv SHARED rwkv.cpp rwkv.h)
else()
    add_library(rwkv rwkv.cpp rwkv.h)
endif()

target_include_directories(rwkv PUBLIC .)
target_compile_features(rwkv PUBLIC cxx_std_11)
target_link_libraries(rwkv PRIVATE ggml ${RWKV_EXTRA_LIBS})

if (RWKV_BUILD_SHARED_LIBRARY)
    set_target_properties(rwkv PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(rwkv PRIVATE RWKV_SHARED RWKV_BUILD)
endif()

if (GGML_CUDA_SOURCES)
    message(STATUS "GGML CUDA sources found, configuring CUDA architecture")
    set_property(TARGET ggml PROPERTY CUDA_ARCHITECTURES OFF)
    set_property(TARGET ggml PROPERTY CUDA_SELECT_NVCC_ARCH_FLAGS "Auto")
    set_property(TARGET rwkv PROPERTY CUDA_ARCHITECTURES OFF)
endif()

if (NOT RWKV_STANDALONE)
    set_property(TARGET ggml PROPERTY GGML_STANDALONE OFF)
    enable_testing()
    add_subdirectory(tests)
    add_subdirectory(extras)
elseif()
    set_property(TARGET ggml PROPERTY GGML_STANDALONE ON)
endif()
