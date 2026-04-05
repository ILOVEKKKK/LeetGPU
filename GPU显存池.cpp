/**
 * @file gpu_mem_pool.h
 * @author Jinjiang li (lijinjiang@baidu.com)
 * @brief
 * @version 0.1
 * @date 2024-09-20
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef INFER_BACKEND_MEMORY_GPU_MEM_POOL_H_
#define INFER_BACKEND_MEMORY_GPU_MEM_POOL_H_
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include <cassert>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "gpu_sched/device_scope.h"
#include "gpu_sched/log.h"
#include "infer/base/macro.h"

namespace adu {
namespace inference {
namespace backend {

struct GpuMemPoolInitOption {
    size_t init_size;
    size_t max_size;
    size_t page_size;
    int device_id;
};

class GpuMemPool {
public:
    void release(CUdeviceptr ptr) {
        {
            std::lock_guard<std::mutex> lock(pool_mutex_);
            CHECK(ptr_to_handles_.find(ptr) != ptr_to_handles_.end()) << "Ptr Don't exit! " << ptr;
            size_t release_pages = 0;
            for (auto& handle_index : ptr_to_handles_[ptr]) {
                if (free_pages_[handle_index] == false) {
                    free_pages_[handle_index] = true;
                    ++release_pages;
                }
            }
            GPU_LOG_DEBUG << "Release page num: " << release_pages
                          << "; Free page num: " << free_page_num_ + release_pages;
            // showPages("release:", ptr);
            free_page_num_ += release_pages;
            cv_.notify_all();
        }
    }

    CUdeviceptr allocate(size_t alloc_size, CUdeviceptr ptr_input = 0, int device_id = 0) {
        CUdeviceptr ptr;
        size_t page_num = (alloc_size + page_size_ - 1) / page_size_;
        CHECK(device_id == device_id_) << "Alloc gpu " << device_id << " mem"
                                       << "But gpu mem pool is on gpu " <<  device_id_;
        {
            adu::perception::gpu_sched::DeviceScope ds(device_id_);
            std::unique_lock<std::mutex> lock(pool_mutex_);
            max_page_num_ = std::max(page_num, max_page_num_);
            if (ptr_input == 0) {
                cuMemAddressReserve(&ptr, page_num * page_size_, 0, 0, 0);
            } else {
                ptr = ptr_input;
            }
            if (ptr_to_handles_.find(ptr) != ptr_to_handles_.end() && checkIfReady(ptr)) {
                for (auto handle_index : ptr_to_handles_[ptr]) {
                    free_pages_[handle_index] = false;
                }
                GPU_LOG_DEBUG << " No need to remmap mem! size: " << ptr_to_handles_[ptr].size()
                              << " free page num: " << free_page_num_ - ptr_to_handles_[ptr].size();
                // showPages("NoMap:", ptr);
                free_page_num_ -= ptr_to_handles_[ptr].size();
                cv_.notify_all();
                return ptr;
            }
            if (page_num > free_page_num_) {
                extendPool(page_num);
            }
            cv_.wait(lock, [this, page_num] { return free_page_num_ >= page_num; });
            cuMemUnmap(ptr, page_num * page_size_);
            ptr_to_handles_[ptr].clear();
            mapMem((CUdeviceptr)ptr, page_num * page_size_, page_num);
            setAccessOnDevice(device_id, ptr, page_num * page_size_);
            GPU_LOG_INFO << "Alloc page num: " << page_num << "; Free page num: " << free_page_num_;
            showPages("Alloc:", ptr);
            cv_.notify_all();
        }
        return ptr;
    }

    bool Init(GpuMemPoolInitOption& init_option) {
        size_t init_size = init_option.init_size;
        size_t max_size = init_option.max_size;
        size_t page_size = init_option.page_size;
        device_id_ = init_option.device_id;
        adu::perception::gpu_sched::DeviceScope ds(device_id_);
        const size_t mb = 1024 * 1024;
        init_size *= mb;
        max_size *= mb;
        page_size *= mb;
        GPU_LOG_INFO << "Gpu mem pool init Begin";
        CHECK(init_size <= max_size) << "Gpu Mem pool init failed: max size < init size";
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_id_;
        size_t granularity;
        CUresult res = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        CHECK(res == CUDA_SUCCESS) << "cuMemGetAllocationGranularity Cuda driver ERROR! Error code: " << res;
        (void)res;
        page_size_ = ((page_size + granularity - 1) / granularity) * granularity;
        size_t init_page_num = (init_size + page_size_ - 1) / page_size_;
        max_page_num_ = (max_size + page_size_ - 1) / page_size_;

        extendPool(init_page_num);
        free_page_num_ = init_page_num;
        total_page_num_ = init_page_num;
        GPU_LOG_INFO << "Gpu mem pool init end";
        return true;
    }

    ~GpuMemPool() {
        for (CUmemGenericAllocationHandle& handle : pages_) {
            cuMemRelease(handle);
        }
    }

    void show(const std::string& title) {
        GPU_LOG_INFO << "========GPU mem pool: " << title << "===========\n"
                     << "========INIT size: " << total_page_num_ * page_size_ / 1024 / 1024 << " MB===========\n"
                     << "========MAX size: " << max_page_num_ * page_size_ / 1024 / 1024 << " MB===========\n"
                     << "========Page size: " << page_size_ / 1024 / 1024 << " MB===========\n";
    }

private:
    bool checkIfReady(CUdeviceptr ptr_input) {
        for (auto handle_index : ptr_to_handles_[ptr_input]) {
            if (!free_pages_[handle_index]) {
                return false;
            }
        }
        return true;
    }

    void showPages(std::string title, CUdeviceptr ptr_input) {
        GPU_LOG_INFO << title << NOFLUSH;
        for (auto handle_index : ptr_to_handles_[ptr_input]) {
            GPU_LOG_INFO << " " << handle_index << NOFLUSH;
        }
        GPU_LOG_INFO << "";
    }

    CUmemGenericAllocationHandle allocatePhysicalMemory(size_t size) {
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_id_;
        CUmemGenericAllocationHandle allocHandle;
        cuMemCreate(&allocHandle, page_size_, &prop, 0);
        return allocHandle;
    }

    void mapMem(CUdeviceptr ptr, size_t alloc_size, size_t page_num) {
        size_t offset = 0;
        size_t map_page_num = 0;
        for (size_t handle_index = 0; handle_index < pages_.size() && map_page_num < page_num; ++handle_index) {
            if (!free_pages_[handle_index]) {
                continue;
            }
            free_pages_[handle_index] = false;
            CUmemGenericAllocationHandle& handle = pages_[handle_index];

            CUresult res = cuMemMap((CUdeviceptr)ptr + offset, page_size_, 0, handle, 0);
            CHECK(res == CUDA_SUCCESS) << "cuMemMap Cuda driver ERROR! Error code: " << res;
            (void)res;
            offset += page_size_;
            ptr_to_handles_[ptr].emplace_back(handle_index);
            map_page_num++;
        }
        CHECK(map_page_num == page_num) << " ERROR! Don't mmap enough pages for alloc";
        free_page_num_ -= page_num;
    }

    void extendPool(size_t page_num) {
        size_t extend_page_num = page_num - free_page_num_;
        if (total_page_num_ + extend_page_num > max_page_num_) {
            extend_page_num = max_page_num_ - total_page_num_;
            total_page_num_ = max_page_num_;
        } else {
            total_page_num_ += extend_page_num;
        }
        if (extend_page_num == 0) {
            return;
        }
        GPU_LOG_INFO << "Extend mem pool with extend size: " << extend_page_nu0m;
        for (size_t i = 0; i < extend_page_num; ++i) {
            CUmemGenericAllocationHandle handle = allocatePhysicalMemory(page_size_);
            pages_.emplace_back(handle);
            free_pages_[(pages_.size() - 1)] = true;
        }
        free_page_num_ += extend_page_num;
    }

    void setAccessOnDevice(int device, CUdeviceptr ptr, size_t size) {
        CUmemAccessDesc accessDesc = {};
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = device;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        // Make the address accessible
        CUresult res = cuMemSetAccess(ptr, size, &accessDesc, 1);
        CHECK(res == CUDA_SUCCESS) << "cuMemSetAccess Cuda driver ERROR! Error code: " << res;
        (void)res;
    }

    size_t free_page_num_ = 0;
    size_t page_size_ = 0;
    size_t total_page_num_ = 0;
    size_t max_page_num_ = 0;
    int device_id_ = 0;
    std::mutex pool_mutex_;
    std::condition_variable cv_;
    std::unordered_map<int, bool> free_pages_;
    std::vector<CUmemGenericAllocationHandle> pages_;
    std::unordered_map<CUdeviceptr, std::vector<int>> ptr_to_handles_;
};

}  // namespace backend
}  // namespace inference
}  // namespace adu
#endif /* INFER_BACKEND_MEMORY_GPU_MEM_POOL_MANGER_H_ */