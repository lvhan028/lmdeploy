// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/array.h"
#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/core/smem.h"
#include <cassert>

namespace turbomind::gemm {

#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 4)
#define L2_CACHEHINT(size) ".L2::" #size "B"
#else
#define L2_CACHEHINT(size)
#endif

template<class T, class Map, class SmemLayout, int Idx>
struct GmemIteratorSm80 {

    using AccessType = Array<T, Map::kAccessC>;
    using Pointer    = get_pointer_type<T>;

    Pointer     smem_;
    const char* src_data_;
    int         src_delta_;

    int src_offset_;
    int dst_offset_;

    int offset_c_;
    int offset_s_;

    int iter_c_{};
    int iter_s_{};

    int src_step_c_;
    int src_step_s_;
    int src_step_k_;

    int stride_s_;
    int smem_offset_ = 0;

    SmemAccessor<T, SmemLayout> smem_data_;

    __device__ GmemIteratorSm80(int stride_s): stride_s_{stride_s}, smem_data_{nullptr}
    {
        int  warp_id = threadIdx.x / WARP_SIZE;
        int  lane_id = threadIdx.x % WARP_SIZE;
        int2 offsets = Map::get_offset(warp_id, lane_id);
        src_offset_  = offsets.x + offsets.y * stride_s_;
        offset_c_    = offsets.x;
        offset_s_    = offsets.y;
        dst_offset_  = SmemLayout::apply(offset_s_, offset_c_);

        src_offset_ *= sizeof(T);
        stride_s_ *= sizeof(T);

        src_step_c_ = sizeof(T) * Map::kDeltaC;
        src_step_s_ = Map::kDeltaS * stride_s_ - sizeof(T) * Map::kIterC * Map::kDeltaC;
    }

    __device__ void SetSmem(Pointer smem)
    {
        smem_      = smem;
        smem_data_ = smem;
    }

    __device__ void ClearSmem(int pipe_iter = 0)
    {
        PRAGMA_UNROLL
        for (int s = 0; s < Map::kIterS; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < Map::kIterC; ++c) {
                Store(&smem_data_(offset_s_ + s * Map::kDeltaS, offset_c_ + c * Map::kDeltaC),
                      Array<T, Map::kAccessC>{});
            }
        }
    }

    template<class Iter>
    __device__ void Prefetch(const Iter& iter, int begin, int count, int)
    {
        // if constexpr (SmemLayout::kIsTrivial) {
        //     if (begin == 0) {
        //         smem_data_.ptr_ += dst_offset_;
        //     }
        // }
        PRAGMA_UNROLL
        for (int s = begin; s < begin + count && s < Map::kIterS; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < Map::kIterC; ++c) {
                auto dst = &smem_data_(offset_s_ + s * Map::kDeltaS, offset_c_ + c * Map::kDeltaC);
                // if constexpr (SmemLayout::kIsTrivial) {
                //     dst = smem_data_.ptr_;
                //     smem_data_.ptr_ += Map::kDeltaC;
                // }
                CpAsync(std::true_type{}, dst, src_data_, static_cast<bool>(iter));
                src_data_ += src_step_c_;
            }
            // if constexpr (SmemLayout::kIsTrivial) {
            //     smem_data_.ptr_ += Map::kDeltaS * SmemLayout::C - Map::kIterC * Map::kDeltaC;
            // }
            src_data_ += src_step_s_;
        }
        // advancing to next stage in a parallel data path is faster
        // src_data_ += src_step_k_;
    }

    template<class Iter>
    __device__ void Prefetch(const Iter& iter)
    {
        auto dst = &smem_data_(offset_s_ + iter_s_ * Map::kDeltaS, offset_c_ + iter_c_ * Map::kDeltaC);
        CpAsync(std::true_type{}, dst, src_data_, static_cast<bool>(iter));

        src_data_ += src_step_c_;
        ++iter_c_;

        if (iter_c_ < Map::kIterC) {
            return;
        }

        iter_c_ = 0;
        src_data_ += src_step_s_;
        ++iter_s_;

        if (iter_s_ < Map::kIterS) {
            return;
        }

        iter_s_ = 0;

        // advancing to next stage in a parallel data path is faster
        // src_data_ += src_step_k_;
    }

    template<class Iter>
    __device__ void Prefetch(const Iter& iter, int pipe_iter)
    {
        Prefetch(iter, 0, Map::kIterS, pipe_iter);
    }

    __device__ void CpAsync(std::true_type, T* dst, const char* __restrict__ src, bool mask)
    {
#if TURBOMIND_ARCH_SM80
        constexpr int size = sizeof(AccessType);
        auto          ptr  = cast_smem_ptr_to_uint(dst);
        // clang-format off
        if constexpr (size == 16) {
            asm volatile("{\n"
                        "  .reg .pred p;\n"
                        "  setp.ne.b32 p, %0, 0;\n"
                        "  @p cp.async.cg.shared.global" L2_CACHEHINT(128) " [%1], [%2], %3;\n"
                        "}\n" ::"r"((int)mask),
                        "r"(ptr),
                        "l"(src),
                        "n"(size));
        } else {
            asm volatile("{\n"
                        "  .reg .pred p;\n"
                        "  setp.ne.b32 p, %0, 0;\n"
                        "  @p cp.async.ca.shared.global" L2_CACHEHINT(128) " [%1], [%2], %3;\n"
                        "}\n" ::"r"((int)mask),
                        "r"(ptr),
                        "l"(src),
                        "n"(size));
        }
        // clang-format on
#else
        assert(TURBOMIND_ARCH_SM80);
#endif
    }

    __device__ void CpAsync(std::false_type, T* dst, const char* __restrict__ src, bool)
    {
#if TURBOMIND_ARCH_SM80
        auto          ptr  = cast_smem_ptr_to_uint(dst);
        constexpr int size = sizeof(AccessType);
        if constexpr (size == 16) {
            asm volatile(
                "cp.async.cg.shared.global" L2_CACHEHINT(128) " [%0], [%1], %2;\n" ::"r"(ptr), "l"(src), "n"(size));
        }
        else {
            asm volatile(
                "cp.async.ca.shared.global" L2_CACHEHINT(128) " [%0], [%1], %2;\n" ::"r"(ptr), "l"(src), "n"(size));
        }
#else
        assert(TURBOMIND_ARCH_SM80);
#endif
    }

    __device__ void Advance(int stages)
    {
        smem_offset_ += SmemLayout::kSize;
        if (smem_offset_ == stages * SmemLayout::kSize) {
            smem_offset_ = 0;
        }
        smem_data_ = smem_ + smem_offset_;
    }

    template<class Iter>
    __device__ void SetTile(const Iter& iter)
    {
        src_data_   = reinterpret_cast<const char*>(iter.OffsetPtr<Idx>(0)) + src_offset_;
        src_step_k_ = iter.template step<Idx>() - Map::kIterS * Map::kDeltaS * stride_s_;
        // src_delta_ = 0;
    }
};

}  // namespace turbomind::gemm