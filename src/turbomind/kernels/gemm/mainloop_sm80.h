#pragma once

#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/gemm/iterator_sm80.h"
#include <cuda_pipeline_primitives.h>

namespace turbomind::gemm {

template<class Impl_>
struct Mainloop_sm80 {

    using Impl = Impl_;

    using T = typename Impl::T;

    using FragC = typename Impl::FragC;

    using SharedStorage = typename Impl::SharedStorage;

    static constexpr int Stages = Impl::Stages;

    using ThreadMapA = typename Impl::ThreadMapA;
    using ThreadMapB = typename Impl::ThreadMapB;

    using SmemLayoutA = typename Impl::SmemLayoutA;
    using SmemLayoutB = typename Impl::SmemLayoutB;

    using GmemIterA = GmemIteratorSm80<T, ThreadMapA, SmemLayoutA, 0>;
    using GmemIterB = GmemIteratorSm80<T, ThreadMapB, SmemLayoutB, 1>;

    static constexpr int kBatchA = (ThreadMapA::kIterS + Impl::ITER_K - 1) / Impl::ITER_K;
    static constexpr int kBatchB = (ThreadMapB::kIterS + Impl::ITER_K - 1) / Impl::ITER_K;

    __device__ void Wait()
    {
        __pipeline_wait_prior(Stages - 2);
        __syncthreads();
    }

    template<class DataIter>
    __device__ void operator()(
        GmemIterA& gmem_A, GmemIterB& gmem_B, FragC& frag_C, DataIter& data_iter, int tile_iter, SharedStorage& storage)
    {
        Impl::SetSmem(gmem_A, gmem_B, storage);

        PRAGMA_UNROLL
        for (int i = 0; i < Stages; ++i) {
            gmem_A.ClearSmem();
            gmem_B.ClearSmem();
            gmem_A.Advance(Stages);
            gmem_B.Advance(Stages);
        }

        __syncthreads();

        PRAGMA_UNROLL
        for (int stage = 0; stage < Stages - 1; ++stage) {
            gmem_A.SetTile(data_iter);
            gmem_B.SetTile(data_iter);
            gmem_A.Prefetch(data_iter, 0);
            gmem_B.Prefetch(data_iter, 0);
            __pipeline_commit();
            gmem_A.Advance(Stages);
            gmem_B.Advance(Stages);
            ++data_iter;
        }

        constexpr bool kFusePrefetch = false;

        typename Impl::StateA state_A{storage};
        typename Impl::StateB state_B{storage};

        auto prefetch = [&](int k) {
            if constexpr (kFusePrefetch) {
                int batch_A = min((k + 1) * kBatchA, ThreadMapA::kIterS) - k * kBatchA;
                int batch_B = min((k + 1) * kBatchB, ThreadMapB::kIterS) - k * kBatchB;
                gmem_A.Prefetch(data_iter, k * kBatchA, batch_A, 0);
                gmem_B.Prefetch(data_iter, k * kBatchB, batch_B, 0);
            }
            if (k == Impl::ITER_K - 1) {
                if constexpr (kFusePrefetch) {
                    ++data_iter;
                    __pipeline_commit();
                }
                Wait();
                gmem_A.smem_data_ = state_A.data;
                gmem_B.smem_data_ = state_B.data;
                state_A.Advance();
                state_B.Advance();
                gmem_A.SetTile(data_iter);
                gmem_B.SetTile(data_iter);
            }
        };

        Wait();

        state_A.Load(0, 0);
        state_B.Load(0, 0);

        gmem_A.SetTile(data_iter);
        gmem_B.SetTile(data_iter);

        if constexpr (kFusePrefetch) {
            prefetch(0);
        }

        PRAGMA_NO_UNROLL
        for (; tile_iter > 0; --tile_iter) {
            if constexpr (!kFusePrefetch) {
                gmem_A.Prefetch(data_iter, 0);
                gmem_B.Prefetch(data_iter, 0);
                ++data_iter;
                __pipeline_commit();
            }
            Impl::Compute(state_A, state_B, frag_C, 0, prefetch);
        }

        __pipeline_commit();
        __pipeline_wait_prior(0);
    }
};

}  // namespace turbomind::gemm