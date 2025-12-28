#include <cuda_runtime.h>

//矩阵乘法算子，计算X与β乘积
__global__ void matrix_mult_kernel(const float* X,const float* beta,float* output,int int n_samples, int n_features)
{
    int tid_x = blockDim.x*blockIdx.x+threadIdx.x;
    int tid_y = blockDim.y*blockIdx.y+threadIdx.y;

    if(tid_x < n_samples && tid_y < n_features)
    {
        float sum = 0.0f;
        for(int i = 0;i < n_samples;++i)
        {
            float X_T_val = X[tid_y*n_samples+i];
            float beta_val = beta[i];
            sum += X_T_val*beta_val;
        }
        output[tid_x] = sum;
    }
}

//sigmoid函数算子
__global__ void sigmoid_kernel(const float *z, float *p, int N)
{
    int global_tid = blockDim.x*blockIdx.x+threadIdx.x;
    if(global_tid < N)
    {
        p[idx] = 1.0f/(1.0f+expf(-z[idx]));
    }
}

//求梯度算子
__global__ void gradient_kernel()

// X, y, beta are device pointers
extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {}
