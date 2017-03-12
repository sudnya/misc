// Author: Sudnya Padalikar
// Date: 01/18/2014
// Brief: vector addition kernel in cuda

#include <stdio.h>

// Kernel that executes on the CUDA device
__global__ void vector_add(float *c, float *a, float *b, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}
 

int main()
{
    float *a_h;
    float *a_d;  // Pointer to host & device arrays
    float *b_h;
    float *b_d;
    float *c_h;
    float *c_d;

    const int N = 10;  // Number of elements in arrays
    size_t bytes = N * sizeof(float);
    
    a_h = (float *)malloc(bytes);        // Allocate array on host
    cudaMalloc((void **) &a_d, bytes);   // Allocate array on device
    
    b_h = (float *)malloc(bytes);        // Allocate array on host
    cudaMalloc((void **) &b_d, bytes);   // Allocate array on device
    
    c_h = (float *)malloc(bytes);        // Allocate array on host
    cudaMalloc((void **) &c_d, bytes);   // Allocate array on device
    
    // Initialize host array and copy it to CUDA device
    for (int i = 0; i < N; i++) {
        a_h[i] = (float)i;
        b_h[i] = (float)i/10;
    }

    cudaMemcpy(a_d, a_h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, bytes, cudaMemcpyHostToDevice);

    // Do calculation on device:
    int block_size = 256;
    int n_blocks = N-1/block_size + 1;

    vector_add <<< n_blocks, block_size >>> (c_d, a_d, b_d, N);
    
    // Retrieve result from device and store it in host array
    cudaMemcpy(c_h, c_d, sizeof(float)*N, cudaMemcpyDeviceToHost);
    
    // Print results
    for (int i=0; i<N; i++) {
        printf("%d %f\n", i, c_h[i]);
    }
    // Cleanup
    free(a_h); 
    cudaFree(a_d);
    free(b_h); 
    cudaFree(b_d);
    free(c_h); 
    cudaFree(c_d);
}

