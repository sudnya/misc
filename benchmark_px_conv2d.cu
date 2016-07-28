/***********************************************************
#
# \file    benchmark_px_conv2d.cu
# \author  Sudnya Padalikar <mailsudnya@gmail.co>
# \date    Sudnya July 3, 2016
# \brief   benchmark for cudnn - with and without per
#          sample filters
***********************************************************/

#include <cudnn.h>
#include <string>
#include <iostream>
#include <vector>
#include <sstream>
#include <stdexcept>

class BenchmarkOptions
{
    public:
        BenchmarkOptions(
            int imageW,
            int imageH,
            int imageChannels,
            int miniBatchSize,

            int filterW,
            int filterH,
            int filters,

            int iterations,
            float alpha,
            float beta,

            bool usePerExampleFilters,

            cudnnConvolutionFwdAlgo_t algorithm)
        :
            imageW(imageW),
            imageH(imageH),
            imageChannels(imageChannels),
            miniBatchSize(miniBatchSize),
            filterW(filterW),
            filterH(filterH),
            filters(filters),
            iterations(iterations),
            alpha(alpha),
            beta(beta),
            usePerExampleFilters(usePerExampleFilters),
            algorithm(algorithm)
        {}

    public:
        int imageW;
        int imageH;
        int imageChannels;
        int miniBatchSize;

        int filterW;
        int filterH;
        int filters;

        int iterations;
        float alpha;
        float beta;

        bool usePerExampleFilters;

        cudnnConvolutionFwdAlgo_t algorithm;

};

void check(cudaError_t status)
{
    if(status != cudaSuccess)
    {
        throw std::runtime_error("CUDA function failed.");
    }
}

void check(cudnnStatus_t status)
{
    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("CUDNN function failed");
    }
}

std::string toString(const BenchmarkOptions& options)
{
    std::stringstream stream;

    stream << "cudnnForwardConvolution(input {" << options.imageH << ", " << options.imageW << ", " << options.imageChannels << ", "
           << options.miniBatchSize << "}, filter {" << options.filterW << ", " << options.filterH << ", " << options.imageChannels << ", " << options.filters << "},"
           << ")";

    return stream.str();
}

void runBenchmark(const BenchmarkOptions& options)
{
    float* source = nullptr;
    float* filter = nullptr;
    float* destination = nullptr;

    int filterMultiplier = 1;
    int miniBatchFactor = options.miniBatchSize;

    if(options.usePerExampleFilters)
    {
        filterMultiplier = options.miniBatchSize;
        miniBatchFactor = 1;
    }

    check(cudaMalloc(&source, sizeof(float) * options.miniBatchSize * options.imageChannels * options.imageH * options.imageW));
    check(cudaMalloc(&destination, sizeof(float) * options.miniBatchSize * options.filters * options.imageH * options.imageW));
    check(cudaMalloc(&filter, sizeof(float) * filterMultiplier * options.filters * options.imageChannels * options.filterH * options.filterW));

    // create descriptors and allocate memory
    cudnnHandle_t handle;

    check(cudnnCreate(&handle));

    cudnnTensorDescriptor_t sourceDescriptor;

    check(cudnnCreateTensorDescriptor(&sourceDescriptor));

    check(cudnnSetTensor4dDescriptor(sourceDescriptor,
                                     CUDNN_TENSOR_NCHW,
                                     CUDNN_DATA_FLOAT,
                                     miniBatchFactor,
                                     options.imageChannels,
                                     options.imageH,
                                     options.imageW));

    cudnnTensorDescriptor_t destinationDescriptor;

    check(cudnnCreateTensorDescriptor(&destinationDescriptor));

    check(cudnnSetTensor4dDescriptor(destinationDescriptor,
                                     CUDNN_TENSOR_NCHW,
                                     CUDNN_DATA_FLOAT,
                                     miniBatchFactor,
                                     options.filters,
                                     options.imageH,
                                     options.imageW));

    cudnnFilterDescriptor_t filterDescriptor;

    check(cudnnCreateFilterDescriptor(&filterDescriptor));


    check(cudnnSetFilter4dDescriptor(filterDescriptor,
                                     CUDNN_DATA_FLOAT, // image data type
                                     CUDNN_TENSOR_NCHW,
                                     options.filters,        // number of output feature maps
                                     options.imageChannels,        // number of input feature maps
                                     options.filterH,        // height of each input filter
                                     options.filterW         // width of  each input fitler
                                     ));

    cudnnConvolutionDescriptor_t convolutionDescriptor;

    check(cudnnCreateConvolutionDescriptor(&convolutionDescriptor));

    check(cudnnSetConvolution2dDescriptor(convolutionDescriptor,
                                          options.filterH / 2,    // zero-padding height
                                          options.filterW / 2,    // zero-padding width
                                          1,        // vertical filter stride
                                          1,        // horizontal filter stride
                                          1, // upscale the input in x-direction
                                          1, // upscale the input in y-direction
                                          CUDNN_CONVOLUTION));

    size_t workspaceSize = 0;

    check(cudnnGetConvolutionForwardWorkspaceSize(handle,
                                                  sourceDescriptor,
                                                  filterDescriptor,
                                                  convolutionDescriptor,
                                                  destinationDescriptor,
                                                  options.algorithm,
                                                  &workspaceSize));

    void* workspace;
    check(cudaMalloc(&workspace, workspaceSize));

    // start timers
    cudaEvent_t start;
    cudaEvent_t end;

    cudaStream_t stream;

    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    check(cudnnSetStream(handle, stream));

    // warm start
    check(cudnnConvolutionForward(handle,
                                  &options.alpha,
                                  sourceDescriptor,
                                  source,
                                  filterDescriptor,
                                  filter,
                                  convolutionDescriptor,
                                  options.algorithm,
                                  workspace,
                                  workspaceSize,
                                  &options.beta,
                                  destinationDescriptor,
                                  destination));

    check(cudaDeviceSynchronize());

    check(cudaEventCreate(&start));
    check(cudaEventCreate(&end));

    check(cudaEventRecord(start, stream));

    // run conv in a loop
    for(size_t i = 0; i < options.iterations; ++i)
    {
        for(size_t j = 0; j < filterMultiplier; ++j)
        {
            size_t inputOffset  = j * (options.imageH * options.imageW * options.imageChannels);
            size_t filterOffset = j * (options.filterH * options.filterW * options.imageChannels * options.filters);
            size_t outputOffset = inputOffset;

            check(cudnnConvolutionForward(handle,
                                          &options.alpha,
                                          sourceDescriptor,
                                          source + inputOffset,
                                          filterDescriptor,
                                          filter + filterOffset,
                                          convolutionDescriptor,
                                          options.algorithm,
                                          workspace,
                                          workspaceSize,
                                          &options.beta,
                                          destinationDescriptor,
                                          destination + outputOffset));
        }
    }

    // stop timers
    check(cudaEventRecord(end, stream));

    check(cudaDeviceSynchronize());

    // print status
    float ms = 0.0f;

    check(cudaEventElapsedTime(&ms, start, end));

    std::cout << toString(options) << " completed in " << (1.e3*ms / options.iterations) << "us\n";

    // free memory

    check(cudaEventDestroy(start));
    check(cudaEventDestroy(end));

    check(cudaStreamDestroy(stream));

    check(cudnnDestroyConvolutionDescriptor(convolutionDescriptor));
    check(cudnnDestroyFilterDescriptor(filterDescriptor));
    check(cudnnDestroyTensorDescriptor(destinationDescriptor));
    check(cudnnDestroyTensorDescriptor(sourceDescriptor));
    check(cudnnDestroy(handle));

    check(cudaFree(workspace));
    check(cudaFree(source));
    check(cudaFree(destination));
    check(cudaFree(filter));
}

std::vector<BenchmarkOptions> createOptions(int miniBatchSize, int iterations)
{
    std::vector<BenchmarkOptions> options;

    options.emplace_back(256, 256, 3, miniBatchSize, 3, 3, 64, iterations, 1.0, 0.0, false, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);
    options.emplace_back(256, 256, 3, miniBatchSize, 3, 3, 64, iterations, 1.0, 0.0, true, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);

    options.emplace_back(128, 128, 64, miniBatchSize, 3, 3, 64, iterations, 1.0, 0.0, false, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);
    options.emplace_back(128, 128, 64, miniBatchSize, 3, 3, 64, iterations, 1.0, 0.0, true, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);

    options.emplace_back(64, 64, 64, miniBatchSize, 3, 3, 128, iterations, 1.0, 0.0, false, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);
    options.emplace_back(64, 64, 64, miniBatchSize, 3, 3, 128, iterations, 1.0, 0.0, true, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);

    options.emplace_back(32, 32, 128, miniBatchSize, 3, 3, 128, iterations, 1.0, 0.0, false, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);
    options.emplace_back(32, 32, 128, miniBatchSize, 3, 3, 128, iterations, 1.0, 0.0, true, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);

    options.emplace_back(16, 16, 128, miniBatchSize, 3, 3, 256, iterations, 1.0, 0.0, false, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);
    options.emplace_back(16, 16, 128, miniBatchSize, 3, 3, 256, iterations, 1.0, 0.0, true, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);

    options.emplace_back(8, 8, 256, miniBatchSize, 3, 3, 256, iterations, 1.0, 0.0, false, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);
    options.emplace_back(8, 8, 256, miniBatchSize, 3, 3, 256, iterations, 1.0, 0.0, true, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);

    options.emplace_back(4, 4, 256, miniBatchSize, 3, 3, 256, iterations, 1.0, 0.0, false, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);
    options.emplace_back(4, 4, 256, miniBatchSize, 3, 3, 256, iterations, 1.0, 0.0, true, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);

    return options;
}

void runBenchmarks()
{
    auto options = createOptions(64, 100);

    for(auto& option : options)
    {
        runBenchmark(option);
    }
}

int main()
{
    runBenchmarks();

    return 0;
}



