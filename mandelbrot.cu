#define MAX_DWELL 256

/** a simple complex type */
struct complex {
	__host__ __device__ complex(float re, float im = 0) {
		this->re = re;
		this->im = im;
	}
	/** real and imaginary part */
	float re, im;
}; // struct complex

// operator overloads for complex numbers                                                       
inline __host__ __device__ complex operator+                                                    
(const complex &a, const complex &b) {                                                          
	return complex(a.re + b.re, a.im + b.im);                                                   
}                                                                                               
inline __host__ __device__ complex operator-                                                    
(const complex &a) { return complex(-a.re, -a.im); }                                            
inline __host__ __device__ complex operator-                                                    
(const complex &a, const complex &b) {                                                          
	return complex(a.re - b.re, a.im - b.im);                                                   
}                                                                                               
inline __host__ __device__ complex operator*                                                    
(const complex &a, const complex &b) {                                                          
	return complex(a.re * b.re - a.im * b.im, a.im * b.re + a.re * b.im);                       
}                                                                                               
inline __host__ __device__ float abs2(const complex &a) {                                       
	return a.re * a.re + a.im * a.im;                                                           
}                                                                                               
inline __host__ __device__ complex operator/                                                    
(const complex &a, const complex &b) {                                                          
	float invabs2 = 1 / abs2(b);                                                                
	return complex((a.re * b.re + a.im * b.im) * invabs2,                                       
								 (a.im * b.re - b.im * a.re) * invabs2);                        
}                                                                                               

/** computes the dwell for a single pixel */                                                    
__device__ int pixel_dwell                                                                      
(int w, int h, complex cmin, complex cmax, int x, int y) {                                      
	complex dc = cmax - cmin;                                                                   
    float fx = (float)x / w, fy = (float)y / h;                                                 
    complex c = cmin + complex(fx * dc.re, fy * dc.im);                                         
    int dwell = 0;                                                                              
    complex z = c;                                                                              
	while(dwell < MAX_DWELL && abs2(z) < 2 * 2) {                                               
        z = z * z + c;                                                                          
        dwell++;                                                                                
    }                                                                                           
    return dwell;                                                                               
}  // pixel_dwell                                                                               

/** computes the dwells for Mandelbrot image                                                    
		@param dwells the output array                                                          
		@param w the width of the output image                                                  
        @param h the height of the output image                                                 
		@param cmin the complex value associated with the left-bottom corner of the             
        image                                                                                   
		@param cmax the complex value associated with the right-top corner of the               
        image                                                                                   
 */                                                                                             
extern "C" __global__                                                                           
void mandelbrot_k                                                                               
(int *dwells, int w, int h, complex cmin, complex cmax) {                                       
	// complex value to start iteration (c)                                                     
	int x = threadIdx.x + blockIdx.x * blockDim.x;                                              
	int y = threadIdx.y + blockIdx.y * blockDim.y;                                              
	int dwell = pixel_dwell(w, h, cmin, cmax, x, y);                                            
	dwells[y * w + x] = dwell;                                                                  
}  // mandelbrot_k                                                                              