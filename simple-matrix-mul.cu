// Author: Sudnya Padalikar
// Date: 01/23/2014
// Brief: simple matrix multiplication kernel in cuda

#include <stdio.h>
#include <cassert>
#include <iostream>


// Kernel that executes on the CUDA device
__global__ void matrixMultiply(float * A, float * B, float * C,
                   int numARows, int numAColumns,
                   int numBRows, int numBColumns,
                   int numCRows, int numCColumns) {
    int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
    //bounds check
    if (rowIdx < numCRows && colIdx < numCColumns) {
        float temp = 0.0;
        for (int i = 0; i < numBRows; ++i) {
            temp += A[(rowIdx*numAColumns)+i]*B[colIdx+(i*numBColumns)];
        }
        C[(rowIdx*numCColumns)+colIdx] = temp;
    }
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

    matrixMultiply <<< num_blocks, block_size >>> (a_d, b_d, c_d, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
    
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


