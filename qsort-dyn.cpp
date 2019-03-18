#include <cuda.h>
#include <nvrtc.h>
#include <vector_types.h>

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

const char *qsortStr = "                                                                        \n\
#define MAX_DEPTH           16                                                                  \n\
#define INSERTION_SORT      32                                                                  \n\
                                                                                                \n\
////////////////////////////////////////////////////////////////////////////////                \n\
// Selection sort used when depth gets too big or the number of elements drops                  \n\
// below a threshold.                                                                           \n\
////////////////////////////////////////////////////////////////////////////////                \n\
__device__                                                                                      \n\
void selection_sort                                                                             \n\
(                                                                                               \n\
    unsigned int *data,                                                                         \n\
    int left,                                                                                   \n\
    int right                                                                                   \n\
)                                                                                               \n\
{                                                                                               \n\
    for (int i = left ; i <= right ; ++i)                                                       \n\
    {                                                                                           \n\
        unsigned min_val = data[i];                                                             \n\
        int min_idx = i;                                                                        \n\
                                                                                                \n\
        // Find the smallest value in the range [left, right].                                  \n\
        for (int j = i+1 ; j <= right ; ++j)                                                    \n\
        {                                                                                       \n\
            unsigned val_j = data[j];                                                           \n\
                                                                                                \n\
            if (val_j < min_val)                                                                \n\
            {                                                                                   \n\
                min_idx = j;                                                                    \n\
                min_val = val_j;                                                                \n\
            }                                                                                   \n\
        }                                                                                       \n\
                                                                                                \n\
        // Swap the values.                                                                     \n\
        if (i != min_idx)                                                                       \n\
        {                                                                                       \n\
            data[min_idx] = data[i];                                                            \n\
            data[i] = min_val;                                                                  \n\
        }                                                                                       \n\
    }                                                                                           \n\
}                                                                                               \n\
                                                                                                \n\
////////////////////////////////////////////////////////////////////////////////                \n\
// Very basic quicksort algorithm, recursively launching the next level.                        \n\
////////////////////////////////////////////////////////////////////////////////                \n\
extern \"C\" __global__                                                                         \n\
void cdp_simple_quicksort                                                                       \n\
(                                                                                               \n\
    unsigned int *data,                                                                         \n\
    int left, int right,                                                                        \n\
    int depth                                                                                   \n\
)                                                                                               \n\
{                                                                                               \n\
    // If we're too deep or there are few elements left, we use an insertion sort...            \n\
    if (depth >= MAX_DEPTH || right-left <= INSERTION_SORT)                                     \n\
    {                                                                                           \n\
        selection_sort(data, left, right);                                                      \n\
        return;                                                                                 \n\
    }                                                                                           \n\
                                                                                                \n\
    unsigned int *lptr = data+left;                                                             \n\
    unsigned int *rptr = data+right;                                                            \n\
    unsigned int  pivot = data[(left+right)/2];                                                 \n\
                                                                                                \n\
    // Do the partitioning.                                                                     \n\
    while (lptr <= rptr)                                                                        \n\
    {                                                                                           \n\
        // Find the next left- and right-hand values to swap                                    \n\
        unsigned int lval = *lptr;                                                              \n\
        unsigned int rval = *rptr;                                                              \n\
                                                                                                \n\
        // Move the left pointer as long as the pointed element is smaller than the pivot.      \n\
        while (lval < pivot)                                                                    \n\
        {                                                                                       \n\
            lptr++;                                                                             \n\
            lval = *lptr;                                                                       \n\
        }                                                                                       \n\
                                                                                                \n\
        // Move the right pointer as long as the pointed element is larger than the pivot.      \n\
        while (rval > pivot)                                                                    \n\
        {                                                                                       \n\
            rptr--;                                                                             \n\
            rval = *rptr;                                                                       \n\
        }                                                                                       \n\
                                                                                                \n\
        // If the swap points are valid, do the swap!                                           \n\
        if (lptr <= rptr)                                                                       \n\
        {                                                                                       \n\
            *lptr++ = rval;                                                                     \n\
            *rptr-- = lval;                                                                     \n\
        }                                                                                       \n\
    }                                                                                           \n\
                                                                                                \n\
    // Now the recursive part                                                                   \n\
    int nright = rptr - data;                                                                   \n\
    int nleft  = lptr - data;                                                                   \n\
                                                                                                \n\
    // Launch a new block to sort the left part.                                                \n\
    if (left < (rptr-data))                                                                     \n\
    {                                                                                           \n\
        cudaStream_t s;                                                                         \n\
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);                                   \n\
        cdp_simple_quicksort<<< 1, 1, 0, s >>>(data, left, nright, depth+1);                    \n\
        cudaStreamDestroy(s);                                                                   \n\
    }                                                                                           \n\
                                                                                                \n\
    // Launch a new block to sort the right part.                                               \n\
    if ((lptr-data) < right)                                                                    \n\
    {                                                                                           \n\
        cudaStream_t s1;                                                                        \n\
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);                                  \n\
        cdp_simple_quicksort<<< 1, 1, 0, s1 >>>(data, nleft, right, depth+1);                   \n\
        cudaStreamDestroy(s1);                                                                  \n\
    }                                                                                           \n\
}                                                                                               \n";

////////////////////////////////////////////////////////////////////////////////
// Initialize data on the host.
////////////////////////////////////////////////////////////////////////////////
void initialize_data(unsigned int *dst, unsigned int nitems)
{
    // Fixed seed for illustration
    srand(2047);

    // Fill dst with random values
    for (unsigned i = 0 ; i < nitems ; i++)
        dst[i] = rand() % nitems ;
}

////////////////////////////////////////////////////////////////////////////////
// Main entry point.
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // cdp only supports sm35 and above
    // see https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications

    // Create an instance of nvrtcProgram with the QSORT code string.
    nvrtcProgram prog;
    NVRTC_SAFE_CALL(
        nvrtcCreateProgram(&prog,           // prog
        qsortStr,                           // buffer
        "qsort.cu",                         // name
        0,                                  // numHeaders
        NULL,                               // headers
        NULL));                             // includeNames

    // Compile the program for compute_35 with relocatable-device-code enabled.
    const char *opts[] =
    {
        "--gpu-architecture=compute_35",
        "--relocatable-device-code=true"
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

    // Load the generated PTX and get a handle to the QSORT kernel.
    CUcontext cuContext;
    CUdevice cuDevice;
    CUmodule cuModule;
    CUfunction cuKernel;

    CUDA_SAFE_CALL(cuInit(0));
    CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
    CUDA_SAFE_CALL(cuCtxCreate(&cuContext, CU_CTX_SCHED_BLOCKING_SYNC | CU_CTX_MAP_HOST, cuDevice));

    CUlinkState linkState;
    void *cubin;
    size_t cubinSize;

    CUDA_SAFE_CALL(cuLinkCreate(0, NULL, NULL, &linkState));
    CUDA_SAFE_CALL(cuLinkAddFile(linkState, CU_JIT_INPUT_LIBRARY, CUDADEVRTLIB, 0, NULL, NULL));
    CUDA_SAFE_CALL(cuLinkAddData(linkState, CU_JIT_INPUT_PTX, (void*)ptx, strlen(ptx), 0, 0, 0, 0));
    CUDA_SAFE_CALL(cuLinkComplete(linkState, &cubin, &cubinSize));
    CUDA_SAFE_CALL(cuModuleLoadData(&cuModule, cubin));
    CUDA_SAFE_CALL(cuModuleGetFunction(&cuKernel, cuModule, "cdp_simple_quicksort"));

    // Generate input for execution, and create output buffers.
    unsigned int num_items = 128;
    unsigned int bufferSize = num_items * sizeof(unsigned int);
    unsigned int *h_Indata = 0;
    unsigned int *h_Outdata = 0;

    h_Indata = (unsigned int *)malloc(bufferSize);
    h_Outdata = (unsigned int *)malloc(bufferSize);
    initialize_data(h_Indata, num_items);

    CUdeviceptr dInOut;
    CUDA_SAFE_CALL(cuMemAlloc(&dInOut, bufferSize));
    CUDA_SAFE_CALL(cuMemcpyHtoD(dInOut, h_Indata, bufferSize));

    int left = 0;
    int right = num_items - 1;
    int depth = 0;

    // Execute QSORT.
    std::cout << "Running qsort-dyn on " << num_items << " elements ..." << std::endl;
    
    void *args[] = { &dInOut, &left, &right, &depth };
    CUDA_SAFE_CALL(
        cuLaunchKernel(cuKernel,
        1, 1, 1,    // grid dim
        1, 1, 1,    // block dim
        0, NULL,    // shared mem and stream
        args, 0));  // arguments
    CUDA_SAFE_CALL(cuCtxSynchronize());

    // Retrieve and check output.
    bool passed = true;
    CUDA_SAFE_CALL(cuMemcpyDtoH(h_Outdata, dInOut, bufferSize));
    for (unsigned int i = 1; i < num_items; ++i) {
        if (h_Outdata[i - 1] > h_Outdata[i]) {
            passed = false;
            break;
        }
    }

    if (passed)
        std::cout << "Passed!" << std::endl;
    else
        std::cout << "Failed!" << std::endl;

    // Release resources.
    CUDA_SAFE_CALL(cuMemFree(dInOut));
    CUDA_SAFE_CALL(cuModuleUnload(cuModule));
    CUDA_SAFE_CALL(cuLinkDestroy(linkState));
    CUDA_SAFE_CALL(cuCtxDestroy(cuContext));

    free(h_Indata);
    free(h_Outdata);
    
    std::cout << std::endl;
    return 0;
}

