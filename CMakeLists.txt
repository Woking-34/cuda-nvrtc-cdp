cmake_minimum_required(VERSION 2.6)

project(cuda-nvrtc-cdp)

SET( CMAKE_VERBOSE_MAKEFILE ON )
IF(NOT CMAKE_BUILD_TYPE)
	SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
ENDIF()

string(TOUPPER ${CMAKE_CXX_COMPILER_ID} CXX_COMPILER_ID_UPPER)
if(${CXX_COMPILER_ID_UPPER} STREQUAL MSVC)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj /MP")
endif()

find_package(CUDA REQUIRED)
IF(CUDA_FOUND)
	MESSAGE(STATUS "CUDA FOUND - ${CUDA_TOOLKIT_ROOT_DIR}")
	MESSAGE(STATUS "CUDA LIBRARY: " ${CUDA_CUDA_LIBRARY})
	MESSAGE(STATUS "CUDA RUNTIME: " ${CUDA_CUDART_LIBRARY})
	
	# find NVRTC - runtime compilation library for CUDA C++
	if (NOT DEFINED CUDA_NVRTC_LIBRARY)
		string(REGEX REPLACE cudart nvrtc CUDA_NVRTC_LIBRARY ${CUDA_CUDART_LIBRARY})
	endif()
	MESSAGE(STATUS "CUDA NVRTC: " ${CUDA_NVRTC_LIBRARY})
	
	# find DEVRT - library to enable separate compilation and linking
	if(CMAKE_SIZEOF_VOID_P EQUAL 8)
		find_library(CUDA_CUDADEVRT_LIBRARY NAMES cudadevrt libcudadevrt PATHS "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
	else()
		find_library(CUDA_CUDADEVRT_LIBRARY NAMES cudadevrt libcudadevrt PATHS "${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32")
	endif()
	MESSAGE(STATUS "CUDA DEVRT: " ${CUDA_CUDADEVRT_LIBRARY})
	
	# find NVTX - tools extension library for marking events and ranges
	if(CMAKE_SIZEOF_VOID_P EQUAL 8)
		find_library(CUDA_NVTX_LIBRARY
		    NAMES nvToolsExt nvToolsExt64_1
		    PATHS "${CUDA_TOOLKIT_TARGET_DIR}/lib" "c:/Program Files/NVIDIA Corporation/NvToolsExt/lib/x64/")
	else()
	    find_library(CUDA_NVTX_LIBRARY
	        NAMES nvToolsExt nvToolsExt32_1
	        PATHS "${CUDA_TOOLKIT_TARGET_DIR}/lib" "c:/Program Files/NVIDIA Corporation/NvToolsExt/lib/Win32/")
	endif()
	find_path(CUDA_NVTX_INCLUDE_DIR nvToolsExt.h
		PATHS "${CUDA_TOOLKIT_TARGET_DIR}/include" "c:/Program Files/NVIDIA Corporation/NvToolsExt/include/")
	MESSAGE(STATUS "CUDA NVTX: " ${CUDA_NVTX_LIBRARY})
ELSE()
	MESSAGE(FATAL_ERROR "CUDA NOT FOUND!")
ENDIF()

add_executable(cuda00_saxpy saxpy.cpp)
target_include_directories(cuda00_saxpy PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(cuda00_saxpy PRIVATE ${CUDA_NVTX_INCLUDE_DIR})
target_link_libraries(cuda00_saxpy ${CUDA_CUDA_LIBRARY})
target_link_libraries(cuda00_saxpy ${CUDA_NVTX_LIBRARY})
target_link_libraries(cuda00_saxpy ${CUDA_NVRTC_LIBRARY})

add_executable(cuda01_qsort-dyn qsort-dyn.cpp)
target_include_directories(cuda01_qsort-dyn PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(cuda01_qsort-dyn PRIVATE ${CUDA_NVTX_INCLUDE_DIR})
target_link_libraries(cuda01_qsort-dyn ${CUDA_CUDA_LIBRARY})
target_link_libraries(cuda01_qsort-dyn ${CUDA_NVTX_LIBRARY})
target_link_libraries(cuda01_qsort-dyn ${CUDA_NVRTC_LIBRARY})
target_compile_definitions(cuda01_qsort-dyn PRIVATE CUDADEVRTLIB="${CUDA_CUDADEVRT_LIBRARY}")

add_executable(cuda02_mandel mandelbrot.cpp)
target_include_directories(cuda02_mandel PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(cuda02_mandel PRIVATE ${CUDA_NVTX_INCLUDE_DIR})
target_link_libraries(cuda02_mandel ${CUDA_CUDA_LIBRARY})
target_link_libraries(cuda02_mandel ${CUDA_NVTX_LIBRARY})
target_link_libraries(cuda02_mandel ${CUDA_NVRTC_LIBRARY})

add_executable(cuda03_mandel-dyn mandelbrot-dyn.cpp)
target_include_directories(cuda03_mandel-dyn PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(cuda03_mandel-dyn PRIVATE ${CUDA_NVTX_INCLUDE_DIR})
target_link_libraries(cuda03_mandel-dyn ${CUDA_CUDA_LIBRARY})
target_link_libraries(cuda03_mandel-dyn ${CUDA_NVTX_LIBRARY})
target_link_libraries(cuda03_mandel-dyn ${CUDA_NVRTC_LIBRARY})
target_compile_definitions(cuda03_mandel-dyn PRIVATE CUDADEVRTLIB="${CUDA_CUDADEVRT_LIBRARY}")

add_executable(cuda04_pinnedmem pinnedmem.cpp)
target_include_directories(cuda04_pinnedmem PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(cuda04_pinnedmem PRIVATE ${CUDA_NVTX_INCLUDE_DIR})
target_link_libraries(cuda04_pinnedmem ${CUDA_CUDA_LIBRARY})
target_link_libraries(cuda04_pinnedmem ${CUDA_NVTX_LIBRARY})
target_link_libraries(cuda04_pinnedmem ${CUDA_NVRTC_LIBRARY})