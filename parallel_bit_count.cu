/***********************************************************
#
# \file    parallel_bit_count.cu
# \author  Sudnya Diamos <mailsudnya@gmail.co>
# \date    Sunday March 5, 2017
# \brief   A program to parallely count the number of 1s in a super big bit array
***********************************************************/

#include <string>
#include <iostream>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <cuda.h>

constexpr int block_size = 1024;
 
__global__ void parallelBitCount(int* ctaScratch, const int *d_in, const int N)
{
    int maxThreads = blockDim.x * gridDim.x;
    int myId = threadIdx.x + blockDim.x * blockIdx.x;

    //int myOnes = __popc(d_in[myId]);
    int myOnes = 0;

    for (int idx = myId; idx < N; idx+=maxThreads) {

        int temp = d_in[idx];
        int totalWord = 8*sizeof(int);

        for (int i = 0; i < totalWord; ++i)
        {
            myOnes += (temp & 1);
            temp    = temp >> 1;
        }
    }

    //std::printf("Thread id %d: my ones %d\n", myId, myOnes);

    // reduce all threads
    __shared__ int bitCounter[block_size];
    bitCounter[threadIdx.x] = myOnes;

    __syncthreads();
    for (int groupStep = 2; groupStep < block_size; groupStep*=2)
    {
        int myTemp       = bitCounter[threadIdx.x];
        int neighborTemp = 0;
        
        if(threadIdx.x % groupStep == 0)
        {
            neighborTemp = bitCounter[threadIdx.x + groupStep/2];
        }

        __syncthreads();
        myTemp += neighborTemp;

        if(threadIdx.x % groupStep == 0)
        {
            bitCounter[threadIdx.x] = myTemp;                      
        }

        __syncthreads();
    }

    int finalCtaCount = bitCounter[0];

    if(threadIdx.x == 0)
    {
        ctaScratch[blockIdx.x] = finalCtaCount;
    }
}

__global__ void serialReduce(int* scratch, int len)
{
    int answer = 0;
    for (int i = 0; i < len; ++i )
    {
        answer += scratch[i];
    }

    scratch[0] = answer;
}

static void check(cudaError_t status)
{
    if(status != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}
 
int main(void)
{
    // host and device pointers for our bit array
    int *aHost, *aDevice;

    // size of bit array
    const int N       = 100;
    size_t arraySize = N * sizeof(int); 

    //alloc on host
    aHost = (int *)malloc(arraySize);
    //alloc on device - always void**
    check(cudaMalloc((void **) &aDevice, arraySize));

    // Initialize host array
    for (int i=0; i<N; i++) 
    {
        aHost[i] = (int)i%2; //TODO: make this random?!
    }

    // copy to device
    check(cudaMemcpy(aDevice, aHost, arraySize, cudaMemcpyHostToDevice));

    // count 1 bits on device
    int n_blocks = (N + block_size - 1)/block_size;

    int *ctaScratch;
    check(cudaMalloc((void **) &ctaScratch, n_blocks*sizeof(int)));
    /*for (int i=0; i<n_blocks; i++) 
    {
        ctaScratch[i] = 0;
    }*/
    
    parallelBitCount <<< n_blocks, block_size >>> (ctaScratch, aDevice, N);
    check(cudaDeviceSynchronize());

    serialReduce <<< 1, 1>>> (ctaScratch, n_blocks);
    check(cudaDeviceSynchronize());
    

    // bring results back to host
    int finalAnswer;
    check(cudaMemcpy(&finalAnswer, ctaScratch, sizeof(int), cudaMemcpyDeviceToHost));
    
    // Print results
    std::cout << "Final answer " << finalAnswer  << "\n";
        
    // Cleanup
    free(aHost); 
    check(cudaFree(aDevice));
}
