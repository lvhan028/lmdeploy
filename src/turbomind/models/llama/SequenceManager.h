// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/models/llama/BlockManager.h"
#include <functional>

namespace turbomind {

struct Sequence {

    /**
     * \brief
     * 状态是怎么迁移的呢？
    */
    enum Status
    {
        kCached = 0,    // 刚创建时，是 kCached的状态
        kLocked,
        kActive
    };

    uint64_t id;
    Status   status = kCached;

    BlockIds  blocks;               // 关联的 kv block ids
    UniqueIds block_unique_ids;

    int input_length = 0;             // 序列中最后一个请求的input_ids的长度

    mutable std::vector<int> tokens;  // update by user。序列中的所有的历史token_id，在推理的时候，根据用户输入的 step，可以调节从哪里开始

    mutable int cache_len = 0;         // 序列中已经 cache 的 token 的数量，包括历史的，以及正在生成的

    // additional data kept round-to-round
    mutable std::vector<std::byte> random_state;  // update by user

    mutable float rope_theta = 0.f;

    // embedding data
    mutable std::vector<std::vector<std::byte>> input_embeddings;
    mutable std::vector<std::pair<int, int>>    input_embedding_ranges;

    explicit Sequence(uint64_t _id): id(_id) {}

    friend std::ostream& operator<<(std::ostream& os, const Sequence& seq);
};

using Sequences = std::vector<const Sequence*>;

inline std::ostream& operator<<(std::ostream& os, const Sequence& seq)
{
    os << "id=" << seq.id << ", status=" << seq.status << ", token_count=" << seq.tokens.size()
       << ", block_count=" << seq.blocks.size() << ", cache_len=" << seq.cache_len
       << ", random_state_size=" << seq.random_state.size();
    return os;
}

class SequenceManager {
public:
    explicit SequenceManager(size_t      layer_num,
                             size_t      head_num,
                             size_t      head_dim,
                             size_t      block_seq_len,
                             double      block_count,
                             int         chunk_size,
                             size_t      elem_bits,
                             int         rank,
                             IAllocator* allocator);

    SequenceManager(const SequenceManager&)     = delete;
    SequenceManager(SequenceManager&&) noexcept = default;

    [[nodiscard]] const Sequence* Create(uint64_t id);

    [[nodiscard]] const Sequence* Get(uint64_t id);

    [[nodiscard]] bool Contains(uint64_t id);

    [[nodiscard]] bool Erase(uint64_t id);

    void UpdateAndSetUnlock(const Sequence& seq);

    struct Outcome {
        int allocation;
        int swap_in;
        int swap_out;
    };

    using AdjustInputCount = std::function<std::pair<int, int>(const Sequences&, const std::vector<int>&)>;

    /**
     * @param sequences:
     * @param context_lengths
     * @param priorities
     * @param step_length
     * @param adjust
     * @return
    */
    [[nodiscard]] Outcome Materialize(Sequences                    sequences,
                                      std::vector<int>             context_lengths,
                                      const std::vector<uint64_t>& priorities,
                                      int                          step_length,
                                      AdjustInputCount             adjust);

    [[nodiscard]] void* GetKeyPtr(int block_id)
    {
        return block_manager_->block(block_id).data;
    }

    [[nodiscard]] void* GetValPtr(int block_id)
    {
        return (std::byte*)GetKeyPtr(block_id) + val_offset_;
    }

    int max_block_count() const noexcept
    {
        return block_manager_->max_block_count();
    }

private:
    void Erase(std::map<uint64_t, Sequence>::iterator& it);

    void CommitUnlockAndFree();

    void VerifyAndLockCached(const Sequences& sequences);

    std::vector<int> CountRequiredBlocks(const Sequences&        sequences,  //
                                         const std::vector<int>& context_lengths,
                                         int                     step_length);

    static void SortByPriority(Sequences&                   sequences,  //
                               std::vector<int>&            context_lengths,
                               const std::vector<uint64_t>& priorities);

    static void AssignAndActivate(const Sequences&        sequences,  //
                                  const std::vector<int>& counts,
                                  const BlockIds&         blocks,
                                  const UniqueIds&        unique_ids);

private:
    int    block_seq_len_;  // 一个 block 存储的 seq 的长度。一个 block 是所有layer的所有 k/v 的长度为seq的cache块
    int    rank_;
    size_t val_offset_{};

    // Use `std::map` to avoid reference invalidation
    std::map<uint64_t, Sequence> sequences_;

    std::unique_ptr<BlockManager> block_manager_;

    BlockIds unlocked_; // ++ lvhan(23.12.26) 被锁住的 block（为啥这些不是在 block_manager_中管呢）。什么时候 unlocked_ 会改变呢？
                        // sequence_end -> Erase -> UpdateAndSetUnlock 被调用时，增加；CommitUnlockAndFree 被调用时 clear；
    BlockIds freed_;    // 被 sequencemgr free出来的 block
};

inline std::ostream& operator<<(std::ostream& os, const SequenceManager::Outcome& oc)
{
    os << "allocation: " << oc.allocation << ", swap-in: " << oc.swap_in << ", swap-out: " << oc.swap_out;
    return os;
}

}  // namespace turbomind
