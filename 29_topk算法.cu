#include <cuda_runtime.h>
#include <algorithm>
using namespace std;

__global__ void top_k_kernel(const float* input, float* output, int N, int k)
{

}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int k) 
{
    sort(input.begin(),input.end());
}
