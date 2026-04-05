#include <cuda_fp16.h>

#include "infer/base/flags.h"
#include "infer/kernel/entry_list.h"

namespace adu {
namespace inference {
namespace kernel {

#define BLOCK_SIZE 32

/**
 * inputs:
 * image_feats:             norm_num x C x H x W //当前帧由普通相机拍摄的图片个数norm_num
 * narrow_image_feats:      narrow_num x C x H x W //当前帧由窄焦距相机拍摄的图片个数narrow_num
 * fisheye_image_feats:     fisheye_num x C x H x W //当前帧由鱼眼相机拍摄的图片个数fisheye_num
 * cams_embeds:             11 x C //11个相机各自的位置嵌入向量，每个相机都有C通道
 * level_embeds:            1 x C //当前层的嵌入向量
 *
 * outputs:
 * lidar2trans              all_num x 3 x 4
 * all_camera_id            all_num
 * feats:                   1 x HW * C //融合后的BEV特征
 */

__global__ void merge_camera_feats(const int C, const int H, const int W, const int norm_num, const int narrow_num,
                                   const int fisheye_num, const float *lidar2image, const float *lidar2narrow,
                                   const float *lidar2fisheye, const float *norm_img_shape,
                                   const float *narrow_pad_shape, const float *norm_camera_id,
                                   const float *narrow_camera_id, const float *fisheye_camera_id,
                                   const float *image_feats, const float *narrow_image_feats,
                                   const float *fisheye_image_feats, const float *cams_embeds,
                                   const float *level_embeds, float *lidar2trans, float *all_img_shape, float *feats) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    if (bz >= norm_num + narrow_num + fisheye_num) {
        return;
    }

    __shared__ float s_feats[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_cams_ebds[BLOCK_SIZE];
    __shared__ float s_level_ebds[BLOCK_SIZE];
    __shared__ int cam_idx;

    // lidar2trans是一个3x4的视角转换矩阵，存储了相机内参与相机外参，这里将三种不同相机的lidar2trans合并为一个整体的视角转换矩阵
    // 只用第一个线程块来做这件事情，避免重复写入
    if (bx == 0 && by == 0 && bz == 0 && tx < 12) {
        if (ty < norm_num) {
            lidar2trans[ty * 12 + tx] = lidar2image[ty * 12 + tx];
        } else if (ty < norm_num + narrow_num) {
            lidar2trans[ty * 12 + tx] = lidar2narrow[(ty - norm_num) * 12 + tx];
        } else if (ty < norm_num + narrow_num + fisheye_num) {
            lidar2trans[ty * 12 + tx] = lidar2fisheye[(ty - norm_num - narrow_num) * 12 + tx];
        }
    }

    // all image shape
    int idx = ty * BLOCK_SIZE + tx;
    if (bz == 0 && bx == 0 && by == 0 && idx < (norm_num + narrow_num + fisheye_num) * 2) {
        if (idx < norm_num * 2) {
            all_img_shape[idx] = norm_img_shape[idx];
        } else if (idx < (norm_num + narrow_num) * 2) {
            all_img_shape[idx] = narrow_pad_shape[idx - norm_num * 2];
        }
    }

    // 根据当前的blockIdx.z计算当前block处理数据来自的camera id
    // 只启动一个block内的一个线程，存入共享内存后广播给整个block
    if (tx == 0 && ty == 0) {
        if (bz < norm_num) {
            cam_idx = static_cast<int>(norm_camera_id[bz]);
        } else if (bz < norm_num + narrow_num) {
            cam_idx = static_cast<int>(narrow_camera_id[bz - norm_num]);
        } else if (bz < norm_num + narrow_num + fisheye_num) {
            cam_idx = static_cast<int>(fisheye_camera_id[bz - norm_num - narrow_num]);
        }
    }
    __syncthreads();

    // 加载数据，包括图像原始特征，相机嵌入以及层嵌入
    int x_idx = bx * BLOCK_SIZE + tx;
    int y_idx = by * BLOCK_SIZE + ty;
    if (bz < norm_num && x_idx < H * W && y_idx < C) {
        s_feats[ty][tx] = image_feats[bz * C * H * W + y_idx * H * W + x_idx];
        //相机和层的嵌入是随着特征通道维度ty变化的，tx代表的是空间(H*W)维度，嵌入向量在空间维度是共享的
        if (tx == 0) {
            s_cams_ebds[ty] = cams_embeds[cam_idx * C + y_idx];
            s_level_ebds[ty] = level_embeds[y_idx];
        }
    } else if (bz < (norm_num + narrow_num) && x_idx < H * W && y_idx < C) {
        s_feats[ty][tx] = narrow_image_feats[(bz - norm_num) * C * H * W + y_idx * H * W + x_idx];
        if (tx == 0) {
            s_cams_ebds[ty] = cams_embeds[cam_idx * C + y_idx];
            s_level_ebds[ty] = level_embeds[y_idx];
        }
    } else if (bz < (norm_num + narrow_num + fisheye_num) && x_idx < H * W && y_idx < C) {
        s_feats[ty][tx] = fisheye_image_feats[(bz - norm_num - narrow_num) * C * H * W + y_idx * H * W + x_idx];
        if (tx == 0) {
            s_cams_ebds[ty] = cams_embeds[cam_idx * C + y_idx];
            s_level_ebds[ty] = level_embeds[y_idx];
        }
    }
    __syncthreads();
    //可优化的地方：交换tx与ty实现合并访存
    y_idx = bx * BLOCK_SIZE + tx;
    x_idx = by * BLOCK_SIZE + ty;
    //转置并写入全局内存
    if (y_idx < H * W && x_idx < C) {
        feats[bz * C * H * W + y_idx * C + x_idx] = s_feats[ty][tx] + s_cams_ebds[ty] + s_level_ebds[ty];
    }
    
    
}

KERNEL_CALLER_DECLARE(merge_camera_feats, const int C, const int H, const int W, const int norm_num,
                      const int narrow_num, const int fisheye_num, const float *lidar2image, const float *lidar2narrow,
                      const float *lidar2fisheye, const float *norm_img_shape, const float *narrow_pad_shape,
                      const float *norm_camera_id, const float *narrow_camera_id, const float *fisheye_camera_id,
                      const float *image_feats, const float *narrow_image_feats, const float *fisheye_image_feats,
                      const float *cams_embeds, const float *level_embeds, float *lidar2trans, float *all_camera_id,
                      float *feats) {
    KERNEL_LAUNCH(merge_camera_feats, C, H, W, norm_num, narrow_num, fisheye_ num, lidar2image, lidar2narrow,
                  lidar2fisheye, norm_img_shape, narrow_pad_shape, norm_camera_id, narrow_camera_id, fisheye_camera_id,
                  image_feats, narrow_image_feats, fisheye_image_feats, cams_embeds, level_embeds, lidar2trans,
                  all_camera_id, feats);
}

__global__ void cam2bev_point_sampling(const int H, const int W, const int D, const int num_cam, const float x0,
                                       const float y0, const float z0, const float dx, const float dy, const float dz,
                                       const float *image_pad_shape, const float *lidar2img,
                                       float *reference_points_cam, int *bev_mask_count) {
    //X和Y代表真实的物理地面BEV网格，Z维度代表不同的相机
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx_z = blockIdx.z * blockDim.z + threadIdx.z;

    // 通过共享内存读取当前相机的视角转换矩阵lidar2img
    volatile __shared__ float lidar2img_sh[3 * 4];
    if (threadIdx.x < 12 && threadIdx.y == 0) {
        int sh_idx = threadIdx.x;
        int num_c = idx_z;
        lidar2img_sh[sh_idx] = lidar2img[12 * num_c + sh_idx];
    }
    __syncthreads();

    // 将3D坐标转换为2D坐标
    if (idx_x < W && idx_y < H && idx_z < num_cam) {
        int ref_index_base = idx_z * H * W + idx_y * W + idx_x;
        //计算当前BEV格子的物理坐标用于做视角转换计算
        float dx_val = idx_x * dx + x0;
        float dy_val = idx_y * dy + y0;

        float scale_x = 1.0 / image_pad_shape[1];
        float scale_y = 1.0 / image_pad_shape[0];
        int count_tmp = 0;
        //BEV特征图中沿着高度方向有D个采样点
        for (int idx_d = 0; idx_d < D; ++idx_d) {
            float dz_val = idx_d * dz + z0;
            //通过视角转换矩阵计算当前BEV点(x,y,z)对应的2D图像位置(u,v,w)
            float x_val =
                lidar2img_sh[0] * dx_val + lidar2img_sh[1] * dy_val + lidar2img_sh[2] * dz_val + lidar2img_sh[3];
            float y_val =
                lidar2img_sh[4] * dx_val + lidar2img_sh[5] * dy_val + lidar2img_sh[6] * dz_val + lidar2img_sh[7];
            float z_val =
                lidar2img_sh[8] * dx_val + lidar2img_sh[9] * dy_val + lidar2img_sh[10] * dz_val + lidar2img_sh[11];

            float eps = 1e-5f;
            x_val = x_val / max(z_val, eps) * scale_x;
            y_val = y_val / max(z_val, eps) * scale_y;
            
            //这里是根据BEV网格索引refindex_base计算具体的存储位置
            int ref_index = ref_index_base * D * 2 + idx_d * 2;
            //将当前BEV格子对应的2D像素位置记录
            reference_points_cam[ref_index] = x_val;
            reference_points_cam[ref_index + 1] = y_val;

            if (z_val > eps && x_val > 0.0f && x_val < 1.0f && y_val > 0.0f && y_val < 1.0f) {
                count_tmp = 1;
            }
        }
        bev_mask_count[ref_index_base] = count_tmp;
    }
}

__global__ void cam2bev_transpose_and_count_v3(const int HW, const int all_cams_num, const int fisheye_cams_num,
                                               const float *bev_mask_count, float *nonfish_count, float *fisheye_count,
                                               float *query_proj) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < HW; i += blockDim.x * gridDim.x) {
        float count_val = 0.f;
        float fisheye_count_val = 0.f;
        for (int j = 0; j < all_cams_num; ++j) {
            float val_c = static_cast<float>(bev_mask_count[j * HW + i]);
            query_proj[i * all_cams_num + j] = val_c;
            if (j < all_cams_num - fisheye_cams_num) {
                count_val += val_c;
            } else {
                fisheye_count_val += val_c;
            }
        }
        nonfish_count[i] = count_val;
        fisheye_count[i] = fisheye_count_val;
    }
}

KERNEL_CALLER_DECLARE(cam2bev_point_sampling, const int H, const int W, const int D, const int num_cam, const float x0,
                      const float y0, const float z0, const float dx, const float dy, const float dz,
                      const float *image_pad_shape, const float *lidar2img, float *reference_points_cam,
                      int *bev_mask_count) {
    KERNEL_LAUNCH(cam2bev_point_sampling, H, W, D, num_cam, x0, y0, z0, dx, dy, dz, image_pad_shape, lidar2img,
                  reference_points_cam, bev_mask_count);
}

KERNEL_CALLER_DECLARE(cam2bev_transpose_and_count_v3, const int HW, const int all_cams_num, const int fisheye_cams_num,
                      const float *bev_mask_count, float *nonfish_count, float *fisheye_count, float *query_proj) {
    KERNEL_LAUNCH(cam2bev_transpose_and_count_v3, HW, all_cams_num, fisheye_cams_num, bev_mask_count, nonfish_count,
                  fisheye_count, query_proj);
}

} /* namespace kernel */
}  // namespace inference
}  // namespace adu