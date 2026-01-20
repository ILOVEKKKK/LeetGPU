#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

// Kernel to compute matrix-vector product: result = X * beta
// X: n_samples x n_features (row-major)
// beta: n_features x 1
// result: n_samples x 1
__global__ void matmul_kernel(const float *X, const float *beta, float *result,
                               int n_samples, int n_features) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_samples) {
    float sum = 0.0f;
    for (int j = 0; j < n_features; j++) {
      sum += X[idx * n_features + j] * beta[j];
    }
    result[idx] = sum;
  }
}

// Kernel to apply sigmoid function: p = 1 / (1 + exp(-z))
__global__ void sigmoid_kernel(const float *z, float *p, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    p[idx] = 1.0f / (1.0f + expf(-z[idx]));
  }
}

// Kernel to compute gradient: gradient = (X^T * (p - y) + lambda * beta) / n_samples
// X: n_samples x n_features (row-major)
// diff: p - y (n_samples x 1)
// gradient: n_features x 1
__global__ void gradient_kernel(const float *X, const float *p, const float *y,
                                 const float *beta, float *gradient,
                                 int n_samples, int n_features, float lambda) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_features) {
    float sum = 0.0f;
    for (int i = 0; i < n_samples; i++) {
      float diff = p[i] - y[i];
      sum += X[i * n_features + idx] * diff;
    }
    gradient[idx] = (sum + lambda * beta[idx]) / float(n_samples);
  }
}

// Kernel to compute diagonal weight matrix: w[i] = p[i] * (1 - p[i])
__global__ void compute_weights_kernel(const float *p, float *w, int n_samples) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_samples) {
    w[idx] = p[idx] * (1.0f - p[idx]);
  }
}

// Kernel to compute Hessian: H = (X^T * W * X) / n_samples + reg * I
// X: n_samples x n_features (row-major)
// W: n_samples x n_samples (diagonal, stored as vector)
// H: n_features x n_features (row-major, symmetric)
__global__ void hessian_kernel(const float *X, const float *w, float *H,
                                int n_samples, int n_features, float reg) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n_features && col < n_features) {
    float sum = 0.0f;
    for (int i = 0; i < n_samples; i++) {
      sum += X[i * n_features + row] * w[i] * X[i * n_features + col];
    }
    sum /= float(n_samples);

    // Add regularization to diagonal
    if (row == col) {
      sum += reg;
    }
    H[row * n_features + col] = sum;
  }
}

// Kernel to update beta: beta = beta - step (Newton's method)
__global__ void newton_update_kernel(float *beta, const float *step, int n_features) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_features) {
    beta[idx] -= step[idx];
  }
}

// CPU function to solve H * step = gradient using Gaussian elimination with partial pivoting
// H is n x n, gradient is n x 1, result stored in step
void solve_linear_system(float *H, float *gradient, float *step, int n) {
  // Create augmented matrix [H | gradient]
  float *aug = new float[n * (n + 1)];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      aug[i * (n + 1) + j] = H[i * n + j];
    }
    aug[i * (n + 1) + n] = gradient[i];
  }

  // Forward elimination with partial pivoting
  for (int k = 0; k < n; k++) {
    // Find pivot
    int pivot_row = k;
    float max_val = fabs(aug[k * (n + 1) + k]);
    for (int i = k + 1; i < n; i++) {
      float val = fabs(aug[i * (n + 1) + k]);
      if (val > max_val) {
        max_val = val;
        pivot_row = i;
      }
    }

    // Swap rows if needed
    if (pivot_row != k) {
      for (int j = 0; j <= n; j++) {
        float temp = aug[k * (n + 1) + j];
        aug[k * (n + 1) + j] = aug[pivot_row * (n + 1) + j];
        aug[pivot_row * (n + 1) + j] = temp;
      }
    }

    // Eliminate below pivot
    for (int i = k + 1; i < n; i++) {
      float factor = aug[i * (n + 1) + k] / aug[k * (n + 1) + k];
      for (int j = k; j <= n; j++) {
        aug[i * (n + 1) + j] -= factor * aug[k * (n + 1) + j];
      }
    }
  }

  // Back substitution
  for (int i = n - 1; i >= 0; i--) {
    step[i] = aug[i * (n + 1) + n];
    for (int j = i + 1; j < n; j++) {
      step[i] -= aug[i * (n + 1) + j] * step[j];
    }
    step[i] /= aug[i * (n + 1) + i];
  }

  delete[] aug;
}

// Kernel to compute L2 norm squared of gradient
__global__ void norm_squared_kernel(const float *gradient, float *partial_sums,
                                     int n_features) {
  extern __shared__ float sdata[];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Load data into shared memory
  sdata[tid] = (idx < n_features) ? gradient[idx] * gradient[idx] : 0.0f;
  __syncthreads();

  // Reduction in shared memory
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // Write result for this block to global memory
  if (tid == 0) {
    partial_sums[blockIdx.x] = sdata[0];
  }
}

extern "C" void solve(const float *X, const float *y, float *beta,
                      int n_samples, int n_features) {
  const int max_iterations = 30;
  const float lambda = 1e-6f;  // Regularization parameter (for gradient)
  const float reg = 1e-6f;     // Regularization for Hessian diagonal
  const int block_size = 256;

  // Initialize beta to zeros
  cudaMemset(beta, 0, n_features * sizeof(float));

  // Allocate temporary device memory
  float *d_z;           // n_samples x 1: X * beta
  float *d_p;           // n_samples x 1: sigmoid(z)
  float *d_w;           // n_samples x 1: weights p*(1-p)
  float *d_gradient;    // n_features x 1
  float *d_H;           // n_features x n_features: Hessian
  float *d_step;        // n_features x 1: Newton step
  float *d_norm_partial; // For norm computation

  cudaMalloc(&d_z, n_samples * sizeof(float));
  cudaMalloc(&d_p, n_samples * sizeof(float));
  cudaMalloc(&d_w, n_samples * sizeof(float));
  cudaMalloc(&d_gradient, n_features * sizeof(float));
  cudaMalloc(&d_H, n_features * n_features * sizeof(float));
  cudaMalloc(&d_step, n_features * sizeof(float));

  int norm_blocks = (n_features + block_size - 1) / block_size;
  cudaMalloc(&d_norm_partial, norm_blocks * sizeof(float));

  // Grid dimensions
  int grid_samples = (n_samples + block_size - 1) / block_size;
  int grid_features = (n_features + block_size - 1) / block_size;

  // Allocate host memory for Hessian and gradient
  float *h_H = new float[n_features * n_features];
  float *h_gradient = new float[n_features];
  float *h_step = new float[n_features];

  // Newton's method loop
  for (int iter = 0; iter < max_iterations; iter++) {
    // 1. Compute z = X * beta
    matmul_kernel<<<grid_samples, block_size>>>(X, beta, d_z, n_samples, n_features);

    // 2. Compute p = sigmoid(z)
    sigmoid_kernel<<<grid_samples, block_size>>>(d_z, d_p, n_samples);

    // 3. Compute gradient = X^T * (p - y) + lambda * beta
    gradient_kernel<<<grid_features, block_size>>>(X, d_p, y, beta, d_gradient,
                                                    n_samples, n_features, lambda);

    // 4. Compute weights w = p * (1 - p)
    compute_weights_kernel<<<grid_samples, block_size>>>(d_p, d_w, n_samples);

    // 5. Compute Hessian H = (X^T * W * X) / n_samples + reg * I
    dim3 grid_hessian(grid_features, grid_features);
    dim3 block_hessian(16, 16);
    hessian_kernel<<<grid_hessian, block_hessian>>>(X, d_w, d_H, n_samples, n_features, reg);

    // 6. Copy Hessian and gradient to host
    cudaMemcpy(h_H, d_H, n_features * n_features * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gradient, d_gradient, n_features * sizeof(float), cudaMemcpyDeviceToHost);

    // 7. Solve H * step = gradient on CPU
    solve_linear_system(h_H, h_gradient, h_step, n_features);

    // 9. Copy step back to device
    cudaMemcpy(d_step, h_step, n_features * sizeof(float), cudaMemcpyHostToDevice);

    // 10. Update beta = beta - step
    newton_update_kernel<<<grid_features, block_size>>>(beta, d_step, n_features);
  }

  // Cleanup
  delete[] h_H;
  delete[] h_gradient;
  delete[] h_step;
  cudaFree(d_z);
  cudaFree(d_p);
  cudaFree(d_w);
  cudaFree(d_gradient);
  cudaFree(d_H);
  cudaFree(d_step);
  cudaFree(d_norm_partial);
}
