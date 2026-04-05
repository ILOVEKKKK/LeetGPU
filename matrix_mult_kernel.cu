#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>
#define BLOCKDIM 32

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", "; \
        std::cerr << "code: " << error << ", reason: " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

__global__ void matrix_mult_smem_kernel(const float* A,const float*B,float* C,int M,int N,int K)
{
    int tid_x = blockDim.x*blockIdx.x+threadIdx.x;
    int tid_y = blockDim.y*blockIdx.y+threadIdx.y;
    float psum = 0.0f;
    __shared__ float smemA[BLOCKDIM][BLOCKDIM];
    __shared__ float smemB[BLOCKDIM][BLOCKDIM];

    int TILE_K = (K+BLOCKDIM-1)/BLOCKDIM;

    for(int i = 0;i < TILE_K;++i)
    {
        //每次加载一个A矩阵的子矩阵与B矩阵的子矩阵进入shared mem，并进行一次矩阵乘法，结果累加到psum上
        //当该线程负责位置的所有子矩阵都完成计算后，得到了最终结果
        if(tid_y < M && BLOCKDIM*i+threadIdx.x < K)
        {
            smemA[threadIdx.y][threadIdx.x] = A[tid_y*K+BLOCKDIM*i+threadIdx.x];
        }
        else
        {
            smemA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if(tid_x < N && BLOCKDIM*i+threadIdx.y < K)
        {
            smemB[threadIdx.y][threadIdx.x] = B[(BLOCKDIM*i+threadIdx.y)*N+tid_x];
        }
        else
        {
            smemB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        for(int s = 0;s < BLOCKDIM;s++)
        {
            psum+= smemA[threadIdx.y][s]*smemB[s][threadIdx.x];
        }
        __syncthreads();
    }
    if(tid_x < N && tid_y < M)
    {
        C[tid_y*N+tid_x] = psum;
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_mult_smem_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

int main() {
    // 1. 定义矩阵维度 (建议使用 BLOCKDIM 的倍数)
    int M = 1024;
    int N = 1024;
    int K = 1024;
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // 2. 分配主机内存并初始化
    std::vector<float> h_A(M * K, 1.0f); // 填充 1.0
    std::vector<float> h_B(K * N, 2.0f); // 填充 2.0
    std::vector<float> h_C(M * N, 0.0f);

    // 3. 分配设备内存
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));

    // 4. 将数据拷贝到显存
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));

    // 5. 调用 solve 函数 (内部会执行你的内核)
    std::cout << "Launching kernel with matrix size: " << M << "x" << N << "..." << std::endl;
    solve(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaGetLastError()); // 捕获核函数启动错误

    // 6. 将结果拷贝回主机
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));

    // 7. 验证结果 (针对 1.0 * 2.0 的矩阵乘法，结果应该是 K * 2.0)
    bool correct = true;
    float expected = (float)K * 1.0f * 2.0f;
    for (int i = 0; i < M * N; i++) {
        if (std::abs(h_C[i] - expected) > 1e-3) {
            correct = false;
            break;
        }
    }

    if (correct) {
        std::cout << "Result is CORRECT!" << std::endl;
    } else {
        std::cout << "Result is WRONG!" << std::endl;
    }

    // 8. 释放资源
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}