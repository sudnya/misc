// Author: Sudnya Padalikar
// Date: 01/26/2014
// Brief: Tiled (into shared memory) matrix multiplication kernel in cuda

#include <stdio.h>
#include <cassert>
#include <iostream>

#define TILE_SIZE 16
// Kernel that executes on the CUDA device

__device__ void tileMultiplyShared(float * A, float * B, float * C,
			             int numARows, int numAColumns,
			             int numBRows, int numBColumns,
			             int numCRows, int numCColumns,
					     int tileAIdX, int tileAIdY,
						 int tileBIdX, int tileBIdY)
{
    //@@ Insert code to implement matrix multiplication here
    //@@ You have to use shared memory for this MP
	float Cvalue = 0.0f;

    __shared__ float ds_A [TILE_SIZE][TILE_SIZE];
    __shared__ float ds_B [TILE_SIZE][TILE_SIZE];
    
    int rowAIdx = tileAIdY + threadIdx.y;
    int colAIdx = tileAIdX + threadIdx.x;
    int rowBIdx = tileBIdY + threadIdx.y;
    int colBIdx = tileBIdX + threadIdx.x;
    
    if (colBIdx < numBColumns && rowBIdx < numBRows) {
        ds_B[threadIdx.y][threadIdx.x] = B[rowBIdx*numBColumns + colBIdx];
	}
	else {
        ds_B[threadIdx.y][threadIdx.x] = 0.0f;
	}

    if (colAIdx < numAColumns && rowAIdx < numARows) {
        ds_A[threadIdx.y][threadIdx.x] = A[rowAIdx*numAColumns + colAIdx];
	}
	else {
        ds_A[threadIdx.y][threadIdx.x] = 0.0f;
	}
	
	__syncthreads();

	if (colBIdx < numCColumns && rowAIdx < numCRows) {
		for (int i = 0; i < TILE_SIZE; ++i) {
			Cvalue += ds_A[threadIdx.y][i] * ds_B[i][threadIdx.x];
		}
		
		C[(threadIdx.y*TILE_SIZE) + threadIdx.x] += Cvalue;
	}
	
	__syncthreads();

}

__device__ void zeroSharedMemory(float* C)
{
	C[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
		
	__syncthreads();
}

__device__ void copyOutSharedMemory(float* C, int numCRows, int numCColumns, float* sC, int startColumn, int startRow)
{
	if((startRow + threadIdx.y < numCRows) && (startColumn + threadIdx.x < numCColumns))
	{
		C[(startRow + threadIdx.y)*numCColumns + (startColumn + threadIdx.x)] = sC[threadIdx.y * TILE_SIZE + threadIdx.x];
	}
}

// Compute C = A * B
__global__ void matrixMultiplyShared(float * A, float * B, float * C,
			             int numARows, int numAColumns,
			             int numBRows, int numBColumns,
			             int numCRows, int numCColumns) {
	
	int tileX = blockIdx.x * TILE_SIZE;
	int tileY = blockIdx.y * TILE_SIZE;
	
	
	// how to call tileMultiplyShared?
	__shared__ float ds_C [TILE_SIZE][TILE_SIZE];
	
	zeroSharedMemory((float*)ds_C);
	
	for(int tileK = 0; tileK < numAColumns; tileK += TILE_SIZE)
	{
		tileMultiplyShared(A, B, (float*)ds_C, numARows, numAColumns, numBRows,
						   numBColumns, numCRows, numCColumns, tileK, tileY, tileX, tileK);
	}
	
	copyOutSharedMemory(C, numCRows, numCColumns, (float*)ds_C, tileX, tileY);
}


void printMatrix(float* M, int rows, int columns) {
    for (int i  = 0; i < rows; ++i) {
        std::cout << "[ ";
        for (int j = 0; j < columns; ++j) {
            std::cout << " " << M[(i*columns) + j];
        }
        std::cout << " ]\n";
    }

    std::cout << "\n";
}

int main()
{
    float *a_h;
    float *a_d;  // Pointer to host & device arrays
    float *b_h;
    float *b_d;
    float *c_h;
    float *c_d;

    int numARows = 3; // number of rows in the matrix A
    int numAColumns = 4; // number of columns in the matrix A
    int numBRows = 4; // number of rows in the matrix B
    int numBColumns = 2; // number of columns in the matrix B
    int numCRows = 3; // number of rows in the matrix C (you have to set this)
    int numCColumns = 2; // number of columns in the matrix C (you have to set this)
    
    size_t a_bytes = numARows * numAColumns * sizeof(float);
    size_t b_bytes = numBRows * numBColumns * sizeof(float);
    size_t c_bytes = numCRows * numCColumns * sizeof(float);
    assert(numAColumns == numBRows);
    assert(numARows == numCRows);
    assert(numBColumns == numCColumns);
 
    a_h = (float *)malloc(a_bytes);        // Allocate array on host
    cudaMalloc((void **) &a_d, a_bytes);   // Allocate array on device
    
    b_h = (float *)malloc(b_bytes);        // Allocate array on host
    cudaMalloc((void **) &b_d, b_bytes);   // Allocate array on device

    // initialize A, B
    for (int i = 0; i < numARows; ++i) {
        for (int j = 0; j < numAColumns; ++j) {
            a_h[(i*numAColumns) + j] = (i*numAColumns) + j;
        }
    }

    printMatrix(a_h, numARows, numAColumns);

    for (int i = 0; i < numBRows; ++i) {
        for (int j = 0; j < numBColumns; ++j) {
            b_h[(i*numBColumns) +j] = (i*numBColumns) + j;
        }
    }
    printMatrix(b_h, numBRows, numBColumns);

    c_h = (float *)malloc(c_bytes);        // Allocate array on host
    cudaMalloc((void **) &c_d, c_bytes);   // Allocate array on device
    
    cudaMemcpy(a_d, a_h, a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, b_bytes, cudaMemcpyHostToDevice);

    // Do calculation on device:
    dim3 block_size = dim3(16, 16, 1);
    dim3 num_blocks = dim3((numCColumns + 16 - 1)/16, (numCRows + 16 - 1)/ 16);

    matrixMultiplyShared <<< num_blocks, block_size >>> (a_d, b_d, c_d, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
    
    // Retrieve result from device and store it in host array
    cudaMemcpy(c_h, c_d, c_bytes, cudaMemcpyDeviceToHost);
    printMatrix(c_h, numCRows, numCColumns);
 
    // Cleanup
    free(a_h); 
    cudaFree(a_d);
    free(b_h); 
    cudaFree(b_d);
    free(c_h); 
    cudaFree(c_d);
}



