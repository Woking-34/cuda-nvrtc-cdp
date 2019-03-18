#include <cuda.h>
#include <nvrtc.h>

#include <cstdio>
#include <iostream>

#define NVRTC_SAFE_CALL(x)                                        \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
      std::cerr << "\nerror: " #x " failed with error "           \
                << nvrtcGetErrorString(result) << '\n';           \
      exit(1);                                                    \
    }                                                             \
  } while(0)
#define CUDA_SAFE_CALL(x)                                         \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      std::cerr << "\nerror: " #x " failed with error "           \
                << msg << '\n';                                   \
      exit(1);                                                    \
    }                                                             \
  } while(0)

#define NUM_THREADS     128
#define NUM_BLOCKS      32

const char *saxpyStr = "                                        \n\
extern \"C\" __global__                                         \n\
void saxpy(float a, float *x, float *y, float *out, size_t n)   \n\
{                                                               \n\
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;           \n\
  if (tid < n) {                                                \n\
    out[tid] = a * x[tid] + y[tid];                             \n\
  }                                                             \n\
}                                                               \n";

int main(int argc, char **argv)
{
    // nvrtc only supports sm20 and above
    // https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications

    // Create an instance of nvrtcProgram with the SAXPY code string.
    nvrtcProgram prog;
    NVRTC_SAFE_CALL(
        nvrtcCreateProgram(&prog,           // prog
        saxpyStr,                           // buffer
        "saxpy.cu",                         // name
        0,                                  // numHeaders
        NULL,                               // headers
        NULL));                             // includeNames

    // Compile the program for compute_35 with fmad disabled.
    const char *opts[] =
    {
        "--gpu-architecture=compute_35",
        "--fmad=false"
    };
    nvrtcResult compileResult = nvrtcCompileProgram(prog,  // prog
                                                    2,     // numOptions
                                                    opts); // options

    // Obtain compilation log from the program.
    size_t logSize;
    NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
    char *log = new char[logSize];
    NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
    if (logSize > 1)
        std::cout << log << '\n';
    delete[] log;
    if (compileResult != NVRTC_SUCCESS) {
        exit(1);
    }

    // Obtain PTX from the program.
    size_t ptxSize;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
    char *ptx = new char[ptxSize];
    NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));

    // Destroy the program.
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

    // Load the generated PTX and get a handle to the SAXPY kernel.
    CUcontext cuContext;
    CUdevice cuDevice;
    CUmodule cuModule;
    CUfunction cuKernel;

    CUDA_SAFE_CALL(cuInit(0));
    CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
    CUDA_SAFE_CALL(cuCtxCreate(&cuContext, CU_CTX_SCHED_BLOCKING_SYNC | CU_CTX_MAP_HOST, cuDevice));

    CUDA_SAFE_CALL(cuModuleLoadDataEx(&cuModule, ptx, 0, 0, 0));
    CUDA_SAFE_CALL(cuModuleGetFunction(&cuKernel, cuModule, "saxpy"));

    // Generate input for execution, and create output buffers.
    size_t num_items = NUM_THREADS * NUM_BLOCKS;
    size_t bufferSize = num_items * sizeof(float);

    float a = 5.1f;
    float *hX = new float[num_items], *hY = new float[num_items], *hOut = new float[num_items];
    for (size_t i = 0; i < num_items; ++i) {
        hX[i] = static_cast<float>(i);
        hY[i] = static_cast<float>(i * 2);
    }

    CUdeviceptr dX, dY, dOut;
    CUDA_SAFE_CALL(cuMemAlloc(&dX, bufferSize));
    CUDA_SAFE_CALL(cuMemAlloc(&dY, bufferSize));
    CUDA_SAFE_CALL(cuMemAlloc(&dOut, bufferSize));
    CUDA_SAFE_CALL(cuMemcpyHtoD(dX, hX, bufferSize));
    CUDA_SAFE_CALL(cuMemcpyHtoD(dY, hY, bufferSize));

    // Execute SAXPY.
    std::cout << "Running saxpy on " << num_items << " elements ..." << std::endl;
    
    void *args[] = { &a, &dX, &dY, &dOut, &num_items };
    CUDA_SAFE_CALL(
        cuLaunchKernel(cuKernel,
        NUM_BLOCKS, 1, 1,    // grid dim
        NUM_THREADS, 1, 1,   // block dim
        0, NULL,             // shared mem and stream
        args, 0));           // arguments
    CUDA_SAFE_CALL(cuCtxSynchronize());

    // Retrieve and check output.
    bool passed = true;
    CUDA_SAFE_CALL(cuMemcpyDtoH(hOut, dOut, bufferSize));
    for (size_t i = 0; i < num_items; ++i) {
        if (a * hX[i] + hY[i] != hOut[i]) {
            passed = false;
            break;
        }
    }

    if (passed)
        std::cout << "Passed!" << std::endl;
    else
        std::cout << "Failed!" << std::endl;

    // Release resources.
    CUDA_SAFE_CALL(cuMemFree(dX));
    CUDA_SAFE_CALL(cuMemFree(dY));
    CUDA_SAFE_CALL(cuMemFree(dOut));
    CUDA_SAFE_CALL(cuModuleUnload(cuModule));
    CUDA_SAFE_CALL(cuCtxDestroy(cuContext));

    delete[] hX;
    delete[] hY;
    delete[] hOut;

    std::cout << std::endl;
    return 0;
}