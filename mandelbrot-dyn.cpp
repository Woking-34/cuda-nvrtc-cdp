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

/** time spent in device */
double gpu_time = 0;

/** gets the color, given the dwell */
void dwell_color(int *r, int *g, int *b, int dwell);

/** save image to disk */
void savePPM(const std::string& name, unsigned char* src, int width, int height, int numChannels)
{
    std::string ext;
    std::string format;

    {
        if (numChannels == 1)
        {
            format = "P5\n";
            ext = ".pgm";
        }
        else if (numChannels == 3 || numChannels == 4)
        {
            format = "P6\n";
            ext = ".ppm";
        }
        else
        {
            //assert(0);
        }

        std::fstream fh((name + ext).c_str(), std::fstream::out | std::fstream::binary);

        fh << format;

        fh << width << " " << height << "\n" << 0xff << std::endl;

        for (int j = 0; j < height; ++j)
        {
            for (int i = 0; i < width; ++i)
            {
                if (numChannels == 1)
                {
                    fh << (unsigned char)(src[numChannels * (i + j*width) + 0]);
                }
                else if (numChannels == 3 || numChannels == 4)
                {
                    fh << (unsigned char)(src[numChannels * (i + j*width) + 0]);
                    fh << (unsigned char)(src[numChannels * (i + j*width) + 1]);
                    fh << (unsigned char)(src[numChannels * (i + j*width) + 2]);
                }
                else
                {
                    //assert(0);
                }
            }
        }

        fh.flush();
        fh.close();
    }
}

#define MAX_DWELL 256

/** block size along */
#define BSX 64
#define BSY 4
/** subdivision when launched from host */
#define INIT_SUBDIV 32

/** gets the color, given the dwell (on host) */
#define CUT_DWELL (MAX_DWELL / 4)
void dwell_color(int *r, int *g, int *b, int dwell) {
	// black for the Mandelbrot set
	if(dwell >= MAX_DWELL) {
		*r = *g = *b = 0;
	} else {
		// cut at zero
		if(dwell < 0)
			dwell = 0;
		if(dwell <= CUT_DWELL) {
			// from black to blue the first half
			*r = *g = 0;
			*b = 128 + dwell * 127 / (CUT_DWELL);
		} else {
			// from blue to white for the second half
			*b = 255;
			*r = *g = (dwell - CUT_DWELL) * 255 / (MAX_DWELL - CUT_DWELL);
		}
	}
}  // dwell_color

/** file path helper */
bool findFullPath(const std::string& root, std::string& filePath)
{
    bool fileFound = false;
    const std::string resourcePath = root;

    filePath = resourcePath + filePath;
    for (unsigned int i = 0; i < 16; ++i)
    {
        std::ifstream file;
        file.open(filePath.c_str());
        if (file.is_open())
        {
            fileFound = true;
            break;
        }

        filePath = "../" + filePath;
    }

    return fileFound;
}

/** data size */
#define W (16 * 1024)
#define H (16 * 1024)

int main(int argc, char **argv)
{
    // Load CUDA C++ source code in character string.
	std::string rootStr = "cuda-nvrtc-cdp/";
    std::string filePath = "mandelbrot-dyn.cu";

    bool fileFound = findFullPath(rootStr, filePath);
    if (fileFound == false)
    {
        std::cout << "CUDA kernel source not found! Exiting ..." << std::endl;

        std::cout << std::endl;
        return -1;
    }

    std::fstream kernelFile(filePath.c_str(), std::ios::in);

    std::stringstream buffer;
    buffer << kernelFile.rdbuf();
    std::string mandelDynStr = buffer.str() + "\n";

    // Create an instance of nvrtcProgram with the MANDEL-DYN code string.
    nvrtcProgram prog;
    NVRTC_SAFE_CALL(
        nvrtcCreateProgram(&prog,           // prog
        mandelDynStr.c_str(),               // buffer
        "mandelDyn.cu",                     // name
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

    // Load the generated PTX and get a handle to the SAXPY kernel.
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
    CUDA_SAFE_CALL(cuModuleGetFunction(&cuKernel, cuModule, "mandelbrot_block_k"));

    // Generate input for execution, and create output buffers.
	int w = W, h = H;
	int dwell_sz = w * h * sizeof(int);
	int *h_dwells = 0;
	
    h_dwells = (int*)malloc(dwell_sz);

    CUdeviceptr dOut;
    CUDA_SAFE_CALL(cuMemAlloc(&dOut, dwell_sz));

    int blockX = BSX;
    int blockY = BSY;
    int gridX = INIT_SUBDIV;
    int gridY = INIT_SUBDIV;
    float2 cmin = make_float2(-1.5f, -1.0f);
    float2 cmax = make_float2(0.5f, 1.0f);
    int x0 = 0;
    int y0 = 0;
    int d = W / INIT_SUBDIV;
    int depth = 0;

    // Execute MANDELBROT-DYN.
    std::cout << "Running mandelbrot-dyn on " << W << " x " << H << " elements" << std::endl;
    double t1 = omp_get_wtime();

    void *args[] = { &dOut, &w, &h, &cmin, &cmax, &x0, &y0, &d, &depth };
    CUDA_SAFE_CALL(
        cuLaunchKernel(cuKernel,
        gridX, gridY, 1,        // grid dim
        blockX, blockY, 1,      // block dim
        0, NULL,                // shared mem and stream
        args, 0));              // arguments
    CUDA_SAFE_CALL(cuCtxSynchronize());

    double t2 = omp_get_wtime();
    gpu_time = t2 - t1;
	
    // Retrieve and save output.
    CUDA_SAFE_CALL(cuMemcpyDtoH(h_dwells, dOut, dwell_sz));

    //for (int i = 0; i < w*h; ++i)
    //{
    //    int curr = h_dwells[i];
    //
    //    int r, g, b;
    //    dwell_color(&r, &g, &b, curr);
    //
    //    int pixel =
    //        (((unsigned int)b) << 0 |
    //        (((unsigned int)g) << 8) |
    //        (((unsigned int)r) << 16));
    //
    //    h_dwells[i] = pixel;
    //}
    //savePPM("mandel-dyn", (unsigned char*)h_dwells, w, h, 4);

	// print performance
	printf("Mandelbrot set computed in %.3lf s, at %.3lf Mpix/s\n", gpu_time, h * w * 1e-6 / gpu_time);

    // Release resources.
    CUDA_SAFE_CALL(cuMemFree(dOut));
    CUDA_SAFE_CALL(cuModuleUnload(cuModule));
    CUDA_SAFE_CALL(cuLinkDestroy(linkState));
    CUDA_SAFE_CALL(cuCtxDestroy(cuContext));

    free(h_dwells);

    std::cout << std::endl;
    return 0;
}