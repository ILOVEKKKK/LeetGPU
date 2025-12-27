#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) 
{
    int num = input_size - kernel_size +1;//输出长度=输入长度-kernel长度+1
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    if(idx < num)
    {
        float sum = 0;
        for(int i = 0;i<kernel_size;i++)//在一个kernel的长度窗口内做乘加运算
        {
            sum += kernel[i]*input[idx+i];
        }
        output[idx] = sum;
    }
    
}

extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}