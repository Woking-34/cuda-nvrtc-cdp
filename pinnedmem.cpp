#include <cuda.h>
#include <nvrtc.h>
#include <vector_types.h>
#include <vector_functions.h>

#include <omp.h>

#include <string>
#include <cstring>
#include <fstream>
#include <sstream>

#include <cstdio>
#include <iomanip>
#include <iostream>

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

void profileCopies(
    float        *h_a,
    float        *h_b,
    CUdeviceptr     d,
    CUstream        s,
    unsigned int    n,
    char         *desc)
{
    printf("\n%s transfers\n", desc);

    unsigned int bytes = n * sizeof(float);

    // events for timing
    CUevent startEvent, stopEvent;

    CUDA_SAFE_CALL(cuEventCreate(&startEvent, CU_EVENT_DEFAULT));
    CUDA_SAFE_CALL(cuEventCreate(&stopEvent, CU_EVENT_DEFAULT));

    CUDA_SAFE_CALL(cuEventRecord(startEvent, 0));
    CUDA_SAFE_CALL(cuMemcpyHtoDAsync(d, h_a, bytes, s));
    CUDA_SAFE_CALL(cuEventRecord(stopEvent, s));
    CUDA_SAFE_CALL(cuEventSynchronize(stopEvent));

    float time;
    CUDA_SAFE_CALL(cuEventElapsedTime(&time, startEvent, stopEvent));
    printf("  Host to Device bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

    CUDA_SAFE_CALL(cuEventRecord(startEvent, 0));
    CUDA_SAFE_CALL(cuMemcpyDtoHAsync(h_b, d, bytes, s));
    CUDA_SAFE_CALL(cuEventRecord(stopEvent, s));
    CUDA_SAFE_CALL(cuEventSynchronize(stopEvent));

    CUDA_SAFE_CALL(cuEventElapsedTime(&time, startEvent, stopEvent));
    printf("  Device to Host bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

    for (int i = 0; i < n; ++i) {
        if (h_a[i] != h_b[i]) {
            printf("*** %s transfers failed ***", desc);
            break;
        }
    }

    // clean up events
    CUDA_SAFE_CALL(cuEventDestroy(startEvent));
    CUDA_SAFE_CALL(cuEventDestroy(stopEvent));
}

int main()
{
    unsigned int nElements = 4 * 1024 * 1024;
    const unsigned int bytes = nElements * sizeof(float);

    // host arrays
    float *h_aPageable, *h_bPageable;
    float *h_aPinned, *h_bPinned;

    // device array
    CUdeviceptr d_a;

    CUcontext cuContext;
    CUdevice cuDevice;
    CUstream cuStream;

    CUDA_SAFE_CALL(cuInit(0));
    CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
    CUDA_SAFE_CALL(cuCtxCreate(&cuContext, CU_CTX_SCHED_BLOCKING_SYNC | CU_CTX_MAP_HOST, cuDevice));
    CUDA_SAFE_CALL(cuStreamCreate(&cuStream, CU_STREAM_DEFAULT));

    // allocate and initialize
    h_aPageable = (float*)malloc(bytes);                            // host pageable
    h_bPageable = (float*)malloc(bytes);                            // host pageable
    CUDA_SAFE_CALL(cuMemHostAlloc((void**)&h_aPinned, bytes, 0));   // host pinned
    CUDA_SAFE_CALL(cuMemHostAlloc((void**)&h_bPinned, bytes, 0));   // host pinned
    CUDA_SAFE_CALL(cuMemAlloc(&d_a, bytes));                        // device

    for (int i = 0; i < nElements; ++i)
    {
        h_aPageable[i] = i;
        h_aPinned[i] = i;
    }

    memset(h_bPageable, 0, bytes);
    memset(h_bPinned, 0, bytes);

    // output device info and transfer size
    printf("Transfer size (MB): %d\n", bytes / (1024 * 1024));

    // perform copies and report bandwidth
    profileCopies(h_aPageable, h_bPageable, d_a, cuStream, nElements, "Pageable");
    profileCopies(h_aPinned, h_bPinned, d_a, cuStream, nElements, "Pinned");

    printf("\n");

    // cleanup
    cuMemFree(d_a);
    cuMemFreeHost(h_aPinned);
    cuMemFreeHost(h_bPinned);

    CUDA_SAFE_CALL(cuStreamDestroy(cuStream));
    CUDA_SAFE_CALL(cuCtxDestroy(cuContext));

    free(h_aPageable);
    free(h_bPageable);

    return 0;
}