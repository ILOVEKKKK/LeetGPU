#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) 
{
    int idx = 4*(blockDim.x*blockIdx.x+threadIdx.x);//每一个线程负责数组中的一个由四个元素组成的颜色反转区间，线程idx是起点
    if(idx < width*height*4)
    {
        for(int i=0;i<3;i++)//处理一个区间内的4个元素
        {
            image[idx+i] = 255-image[idx+i];
        }
    }
}

extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}