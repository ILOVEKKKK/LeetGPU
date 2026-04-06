#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// ---------------------------------------------------------
// 1. 宏定义与参数设置 (为了教学方便，使用固定的 Head Dim 和 Block Size)
// ---------------------------------------------------------
#define HEAD_DIM 64      // d: 隐藏层维度
#define BLOCK_SIZE_R 32  // Br: Q 的分块大小
#define BLOCK_SIZE_C 32  // Bc: K, V 的分块大小

// 错误检查宏
#define CHECK_CUDA(call) \
do { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA Error at line " << __LINE__ << ": " \
                  << cudaGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ---------------------------------------------------------
// 2. Flash Attention v1 CUDA Kernel
// ---------------------------------------------------------
__global__ void flash_attention_v1_kernel(
    const float* Q, const float* K, const float* V, float* O,
    int N, float scale) 
{
    // 获取当前的 batch 和 head 索引
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    
    // 获取 Q 的分块索引和线程索引
    int q_block_idx = blockIdx.x;
    int tid = threadIdx.x; // 0 到 BLOCK_SIZE_R - 1

    // 计算当前 batch 和 head 的数据偏移量
    // 形状: (Batch, Heads, SeqLen, HeadDim)
    int batch_head_offset = (batch_idx * gridDim.y + head_idx) * N * HEAD_DIM;

    // 当前线程负责的全局 Q 行号
    int global_q_row = q_block_idx * BLOCK_SIZE_R + tid;

    // 申请共享内存存储 K 和 V 的分块
    __shared__ float s_K[BLOCK_SIZE_C][HEAD_DIM];
    __shared__ float s_V[BLOCK_SIZE_C][HEAD_DIM];

    // 线程局部寄存器：存储自己负责的 Q 行、输出 O 行以及统计量
    float q_row[HEAD_DIM];       
    float o_row[HEAD_DIM] = {0}; 
    float m_i = -1e20f;          // 对应公式中的 m (max)
    float l_i = 0.0f;            // 对应公式中的 l (sum)

    // 将 Q_i 加载到寄存器 (注意边界检查)
    if (global_q_row < N) {
        for (int d = 0; d < HEAD_DIM; d++) {
            q_row[d] = Q[batch_head_offset + global_q_row * HEAD_DIM + d];
        }
    }

    // 外层循环：遍历 K 和 V 的分块
    int num_k_blocks = (N + BLOCK_SIZE_C - 1) / BLOCK_SIZE_C;
    for (int k_block_idx = 0; k_block_idx < num_k_blocks; k_block_idx++) {
        
        // --- 步骤 A: 协同加载 K 和 V 分块到共享内存 ---
        int total_elements = BLOCK_SIZE_C * HEAD_DIM;
        for (int i = tid; i < total_elements; i += blockDim.x) {
            int r = i / HEAD_DIM;
            int c = i % HEAD_DIM;
            int global_k_row = k_block_idx * BLOCK_SIZE_C + r;
            
            if (global_k_row < N) {
                s_K[r][c] = K[batch_head_offset + global_k_row * HEAD_DIM + c];
                s_V[r][c] = V[batch_head_offset + global_k_row * HEAD_DIM + c];
            } else {
                s_K[r][c] = 0.0f; // 越界部分补零
                s_V[r][c] = 0.0f;
            }
        }
        __syncthreads(); // 等待当前 Block 的 K, V 加载完毕

        // --- 步骤 B: 计算注意力分数并执行 Online Softmax ---
        if (global_q_row < N) {
            float s_ij[BLOCK_SIZE_C];
            float m_ij = -1e20f; // 当前分块的局部最大值

            // 1. 计算 S_ij = (Q_i * K_j^T) * scale
            for (int j = 0; j < BLOCK_SIZE_C; j++) {
                float sum = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) {
                    sum += q_row[d] * s_K[j][d];
                }
                sum *= scale;
                s_ij[j] = sum;
                if (sum > m_ij) m_ij = sum;
            }

            // 2. Online Softmax 核心逻辑
            float m_new = max(m_i, m_ij);
            float exp_old = expf(m_i - m_new); // 用于缩放旧的输出和累加和
            
            float l_ij = 0.0f;
            for (int j = 0; j < BLOCK_SIZE_C; j++) {
                float p = expf(s_ij[j] - m_new);
                s_ij[j] = p; // 原地保存 P_ij
                l_ij += p;
            }
            
            float l_new = exp_old * l_i + l_ij;

            // 3. 更新输出 O_i = O_i * exp_old + P_ij * V_j
            for (int d = 0; d < HEAD_DIM; d++) {
                float pv_sum = 0.0f;
                for (int j = 0; j < BLOCK_SIZE_C; j++) {
                    pv_sum += s_ij[j] * s_V[j][d];
                }
                o_row[d] = exp_old * o_row[d] + pv_sum;
            }

            // 4. 更新统计量，进入下一个分块
            m_i = m_new;
            l_i = l_new;
        }
        __syncthreads(); // 必须同步，防止有线程跑得快修改了 shared memory
    }

    // --- 步骤 C: 除以最终的累加和 l_i，并写回显存 ---
    if (global_q_row < N) {
        for (int d = 0; d < HEAD_DIM; d++) {
            O[batch_head_offset + global_q_row * HEAD_DIM + d] = o_row[d] / l_i;
        }
    }
}

// ---------------------------------------------------------
// 3. CPU 基准验证 (Standard Attention)
// ---------------------------------------------------------
void standard_attention_cpu(
    const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V, 
    std::vector<float>& O_ref, int B, int H, int N, float scale) 
{
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            int offset = (b * H + h) * N * HEAD_DIM;
            
            for (int i = 0; i < N; i++) {
                std::vector<float> scores(N, 0.0f);
                float max_score = -1e20f;
                
                // Q * K^T
                for (int j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (int d = 0; d < HEAD_DIM; d++) {
                        sum += Q[offset + i * HEAD_DIM + d] * K[offset + j * HEAD_DIM + d];
                    }
                    scores[j] = sum * scale;
                    if (scores[j] > max_score) max_score = scores[j];
                }
                
                // Softmax
                float exp_sum = 0.0f;
                for (int j = 0; j < N; j++) {
                    scores[j] = std::exp(scores[j] - max_score);
                    exp_sum += scores[j];
                }
                for (int j = 0; j < N; j++) {
                    scores[j] /= exp_sum;
                }
                
                // P * V
                for (int d = 0; d < HEAD_DIM; d++) {
                    float out_val = 0.0f;
                    for (int j = 0; j < N; j++) {
                        out_val += scores[j] * V[offset + j * HEAD_DIM + d];
                    }
                    O_ref[offset + i * HEAD_DIM + d] = out_val;
                }
            }
        }
    }
}

// ---------------------------------------------------------
// 4. Main 函数执行与验证
// ---------------------------------------------------------
int main() {
    // 设置参数
    const int B = 2;        // Batch size
    const int H = 4;        // Number of heads
    const int N = 256;      // Sequence length
    const float scale = 1.0f / std::sqrt(HEAD_DIM);
    
    const int total_elements = B * H * N * HEAD_DIM;
    const size_t bytes = total_elements * sizeof(float);

    // Host 内存分配与初始化
    std::vector<float> h_Q(total_elements), h_K(total_elements), h_V(total_elements);
    std::vector<float> h_O_cpu(total_elements, 0.0f), h_O_gpu(total_elements, 0.0f);

    // 随机初始化数据
    for (int i = 0; i < total_elements; ++i) {
        h_Q[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        h_K[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        h_V[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }

    // Device 内存分配
    float *d_Q, *d_K, *d_V, *d_O;
    CHECK_CUDA(cudaMalloc(&d_Q, bytes));
    CHECK_CUDA(cudaMalloc(&d_K, bytes));
    CHECK_CUDA(cudaMalloc(&d_V, bytes));
    CHECK_CUDA(cudaMalloc(&d_O, bytes));

    // 拷贝数据到 Device
    CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), bytes, cudaMemcpyHostToDevice));

    // 配置 Kernel 执行参数
    int grid_x = (N + BLOCK_SIZE_R - 1) / BLOCK_SIZE_R;
    dim3 grid(grid_x, H, B);
    dim3 block(BLOCK_SIZE_R); // 32 个线程，每个线程处理 Q 的一行

    std::cout << "Launching Flash Attention Kernel..." << std::endl;
    flash_attention_v1_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, N, scale);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 拷贝结果回 Host
    CHECK_CUDA(cudaMemcpy(h_O_gpu.data(), d_O, bytes, cudaMemcpyDeviceToHost));

    // 运行 CPU 版本作为基准
    std::cout << "Running CPU Standard Attention for validation..." << std::endl;
    standard_attention_cpu(h_Q, h_K, h_V, h_O_cpu, B, H, N, scale);

    // 验证结果误差
    float max_diff = 0.0f;
    for (int i = 0; i < total_elements; ++i) {
        float diff = std::abs(h_O_cpu[i] - h_O_gpu[i]);
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << "Max Absolute Error between CPU and GPU Flash Attention: " << max_diff << std::endl;
    if (max_diff < 1e-4) {
        std::cout << "=> SUCCESS! The CUDA Kernel calculates Attention correctly." << std::endl;
    } else {
        std::cout << "=> FAILED! The error is too large." << std::endl;
    }

    // 释放内存
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);

    return 0;
}