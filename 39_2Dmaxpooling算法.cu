#include <cuda_runtime.h>
#include <float.h>

__global__ void max_pooling_kernel(const float* input, float* output, int N, int C, int H, int W,
                      int kernel_size, int stride, int padding)
{
    int tid_x = blockDim.x*blockIdx.x+threadIdx.x;
    int tid_y = blockDim.y*blockIdx.y+threadIdx.y;
    int tid_z = blockIdx.z;
    //N和C相乘构成z维度，因此需要获取当前线程所在的batch编号与channel编号
    int n = tid_z/N;
    int c = tid_z%C;

    int H_out = (H+2*padding-kernel_size)/stride+1;
    int W_out = (W+2*padding-kernel_size)/stride+1;

    int global_tid = n*C*H_out*W_out+c*H_out*W_out+tid_y*W_out+tid_x;

    if(tid_x < W_out && tid_y < H_out)
    {
        //计算原始图像中输入窗口的左上角坐标
        int cur_pooling_x = tid_x*stride-padding;
        int cur_pooling_y = tid_y*stride-padding;

        float max_val = -FLT_MAX;
        for(int i = 0;i<kernel_size;++i)
        {
            for(int j = 0;j<kernel_size;++j)
            {
                int pooling_x = cur_pooling_x + i;
                int pooling_y = cur_pooling_y + j;
                if(pooling_x < W && pooling_x >=0 && pooling_y >=0 && pooling_y < H)
                {
                    int input_idx = n*(C*H*W)+c*(H*W)+pooling_y*W+pooling_x;
                    max_val = fmaxf(max_val,input[input_idx]);
                }
            }
        }
        output[global_tid] = max_val;
    }

}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N, int C, int H, int W,
                      int kernel_size, int stride, int padding) 
{
    dim3 threadsPerBlock(16,16,1);
    dim3 blocksPerGrid((W+threadsPerBlock.x-1)/threadsPerBlock.x,(H+threadsPerBlock.y-1)/threadsPerBlock.y,(N*C+threadsPerBlock.z-1)/threadsPerBlock.z);
    max_pooling_kernel<<<blocksPerGrid,threadsPerBlock>>>(input,output,N,C,H,W,kernel_size,stride,padding);
    cudaDeviceSynchronize();
}
