// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/core/data_type.h"
#include "src/turbomind/kernels/gemm/gemm_universal.h"
#include "src/turbomind/kernels/gemm/kernel.h"

namespace turbomind::gemm {

template<class Gemm>
class KernelImpl: public Kernel {
public:
    // import frequently used constants
    static constexpr int CTA_M = Gemm::CTA_M;
    static constexpr int CTA_N = Gemm::CTA_N;
    static constexpr int CTA_K = Gemm::CTA_K;

    using Impl = typename Gemm::Impl;

    KernelImpl()
    {
        desc_.order_a = LayoutType::kRowMajor;
        desc_.order_b = LayoutType::kFragment_81616;
        desc_.order_c = LayoutType::kRowMajor;

        desc_.type_a = get_data_type_v<typename Gemm::T>;
        desc_.type_b = get_data_type_v<typename Gemm::Tb>;
        desc_.type_c = get_data_type_v<typename Gemm::T>;

        desc_.quant_b = QuantDesc{QuantType::kAsym_FMA, Impl::G};

        desc_.cta_tile  = {Gemm::CTA_M, Gemm::CTA_N, Gemm::CTA_K};
        desc_.warp_tile = {Impl::WARP_M, Impl::WARP_N, Impl::WARP_K};
        chunk_size_k_   = Gemm::kChunkSizeK;

        desc_.align_m = Gemm::AlignedM;
        desc_.align_n = Gemm::AlignedN;

        smem_size_ = sizeof(typename Gemm::SharedStorage);

        desc_.stages  = Impl::Stages;
        desc_.split_k = Gemm::SplitK;
        desc_.swizzle = Gemm::CtaMap::N;

        desc_.arch = 80;

        if (smem_size_ > (48 << 10)) {
            cudaFuncSetAttribute(gemm_kernel<Gemm>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_);
        }

        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_ctas_, gemm_kernel<Gemm>, Impl::WARP_CNT * WARP_SIZE, smem_size_);

        name_ = GetName();
    }

    int Launch(const Operation&    operation,
               const void*         alpha,
               const void*         A,
               const MatrixLayout& Adesc,
               const void*         B,
               const MatrixLayout& Bdesc,
               const void*         Q,
               const MatrixLayout& Qdesc,
               const void*         beta,
               const void*         C,
               const MatrixLayout& Cdesc,
               void*               D,
               const MatrixLayout& Ddesc,
               int                 splits,
               void*               barriers,
               size_t&             barriers_size,
               void*               workspace,
               size_t&             workspace_size,
               cudaStream_t        stream) override
    {
        using Map = typename Gemm::CtaMap;

        const int m = Ddesc.rows;
        const int n = Ddesc.cols;
        const int k = Adesc.cols;

        const auto tiles = Map::get_tiled_shape(m, n, k, CTA_M, CTA_N, splits);

        if (splits > 1 && barriers == nullptr && workspace == nullptr) {
            GetWorkspaceSizes(m, n, tiles.x, tiles.y, splits, barriers_size, workspace_size);
            return -1;
        }

        const auto log_tile = Map::get_log_tile(tiles);

        const auto grid  = Map::get_grid_shape(tiles);
        const auto block = Gemm::Impl::WARP_CNT * WARP_SIZE;

        using Ta = typename Gemm::T;
        using Tb = typename Gemm::Tb;
        using Tq = typename Gemm::Tq;
        using Tc = typename Gemm::T;

        [[maybe_unused]] static const int _ = [] {
            std::cout << "A:\n";
            Print(typename Impl::ThreadMapA{});
            std::cout << "\nB:\n";
            Print(typename Impl::ThreadMapB{});
            std::cout << "\nQ:\n";
            Print(typename Impl::ThreadMapQ{});
            printf("warp count: %d\n", Impl::WARP_CNT);
            return 0;
        }();

        typename Gemm::Param param{
            (Ta*)A, _cast((Tb*)B), (Tq*)Q, (Tc*)C, m, n, k, log_tile, tiles, (float*)workspace, (int*)barriers};

        gemm_kernel<Gemm><<<grid, block, smem_size_, stream>>>(param, Map{});

        return 0;
    }

    template<class T>
    static auto _cast(T* p)
    {
        if constexpr (bitsof<T> % 8 == 0) {
            return p;
        }
        else {
            return (char*)p;
        }
    }

    // ! This assumes n results is 16 byte aligned partials
    void
    GetWorkspaceSizes(int m, int n, int tiled_m, int tiled_n, int splits, size_t& barriers_size, size_t& workspace_size)
    {
        workspace_size = sizeof(float) * m * n * splits;
        barriers_size  = sizeof(int) * tiled_m * tiled_n * splits;
    }

    int GetMaxSplits(int m, int n, size_t barrier_size, size_t workspace_size) override
    {
        if (!Gemm::SplitK) {  // kernel has no split-k support
            return 1;
        }

        const int tiled_m = ceil_div(m, CTA_M);
        const int tiled_n = ceil_div(m, CTA_N);

        size_t barriers_per_split{};
        size_t workspace_per_split{};

        // workspace for 1 non-trival split
        GetWorkspaceSizes(m, n, tiled_m, tiled_n, 1, barriers_per_split, workspace_per_split);

        return std::min<int>(barrier_size / barriers_per_split, workspace_size / workspace_per_split);
    }
};

}  // namespace turbomind::gemm