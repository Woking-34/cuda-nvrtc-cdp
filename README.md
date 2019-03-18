# CUDA NVRTC & CDP samples

### Feature list:
 * CUDA Driver API
 * CUDA Runtime Compilation
 * CUDA Dynamic Parallelism
 * saxpy, qsort, mandelbrot samples

### Requirements
 * CMake
 * NVRTC requires CUDA GPU sm20+
 * CDP requires CUDA GPU sm35+
 * Windows only

<p align="center"><img src="mandelbrot.jpg" width="800" /></p>
<p align="center"><b>Mandelbrot set with CUDA Dynamic Parallelism - The Mariani-Silver Algorithm</b></p>

### References:
 * [NVIDIA CUDA SDK samples](https://github.com/NVIDIA/cuda-samples)
 * [NVRTC saxpy sample - CUDA Toolkit Documentation](http://docs.nvidia.com/cuda/nvrtc/index.html#code-saxpy-cpp)
 * [NVRTC cdp sample - CUDA Toolkit Documentation](http://docs.nvidia.com/cuda/nvrtc/index.html#example-dynamic-parallelism)
 * [Adaptive Parallel Computation with CDP - Parallel Forall](https://devblogs.nvidia.com/parallelforall/introduction-cuda-dynamic-parallelism/)
 * [CUDA Dynamic Parallelism API and Principles - Parallel Forall](https://devblogs.nvidia.com/parallelforall/cuda-dynamic-parallelism-api-principles/)
