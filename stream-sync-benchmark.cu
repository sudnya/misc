
#include <vector>
#include <cassert>
#include <iostream>

__device__ int dummy;

__global__ void dummy_kernel()
{
    dummy = 0;
}

void check(cudaError_t status)
{
    assert(status == cudaSuccess);
}

void benchmark(int streamCount, int kernelsPerStream)
{
    std::vector<cudaStream_t> streams;
    std::vector<cudaEvent_t> beginEvents;
    std::vector<cudaEvent_t> endEvents;

    for(int i = 0; i < streamCount; ++i)
    {
        streams.push_back(nullptr);
        check(cudaStreamCreateWithFlags(&streams.back(), cudaStreamNonBlocking));
    }

    for(int i = 0; i < streamCount; ++i)
    {
        endEvents.push_back(nullptr);
        check(cudaEventCreate(&endEvents.back()));
        beginEvents.push_back(nullptr);
        check(cudaEventCreate(&beginEvents.back()));
    }

    for(int i = 0; i < streamCount; ++i)
    {
        check(cudaEventRecord(beginEvents[i], streams[i]));
    }

    for(int i = 0; i < kernelsPerStream; ++i)
    {
        for(int j = 0; j < streamCount; ++j)
        {
            dummy_kernel<<<1, 1, 0, streams[j]>>>();
        }
    }

    for(int i = 0; i < streamCount; ++i)
    {
        check(cudaEventRecord(endEvents[i], streams[i]));
    }

    check(cudaDeviceSynchronize());

    double duration = 0.0;

    for(int i = 0; i < streamCount; ++i)
    {
        float ms = 0.0f;

        check(cudaEventElapsedTime(&ms, beginEvents[i], endEvents[i]));

        duration = std::max(static_cast<double>(ms), duration);
    }

    std::cout << "Overhead per kernel is " << (duration * 1e3) / (streamCount * kernelsPerStream) << "us\n";

    for(auto event : beginEvents)
    {
        check(cudaEventDestroy(event));
    }

    for(auto event : endEvents)
    {
        check(cudaEventDestroy(event));
    }

    for(auto stream : streams)
    {
        check(cudaStreamDestroy(stream));
    }

}

int main()
{
    benchmark(1, 100);

    benchmark(1, 50*100000);
    benchmark(50, 100000);

    return 0;
}



