// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/SequenceManager.h"
#include "src/turbomind/models/llama/BlockManager.h"
#include "src/turbomind/utils/allocator.h"
#include "src/turbomind/utils/debug_utils.h"
#include "src/turbomind/utils/logger.h"
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include <stdexcept>

namespace turbomind {

SequenceManager::SequenceManager(size_t      layer_num,
                                 size_t      head_num, // kv head number
                                 size_t      head_dim,
                                 size_t      block_seq_len,
                                 double      block_count,
                                 int         chunk_size,
                                 size_t      elem_bits,
                                 int         rank,
                                 IAllocator* allocator):
    block_seq_len_(block_seq_len),
    rank_(rank)
{
    constexpr int kBitsPerByte = 8;

    // [2, L, H, block_seq_len, D]
    size_t block_size = 2UL * layer_num * head_num * block_seq_len * head_dim * elem_bits / kBitsPerByte;

    block_manager_ = std::make_unique<BlockManager>(block_size, block_count, chunk_size, allocator);

    val_offset_ = block_size / 2;
}

/**
 * \brief 创建新的序列
 * \param [in] id 序列的id
 * \return
 * \note 当request的start_flag == true的时候，调用 Create。这时候表示序列的第一轮请求。
*/
const Sequence* SequenceManager::Create(uint64_t id)
{
    Sequence sequence{id};
    auto     it = sequences_.find(id);
    if (it != sequences_.end()) {
        if (rank_ == 0) {
            TM_LOG_WARNING("[SequenceManager][Create] Removing conflicting ID %ld", (long)id);
        }
        Erase(it);
    }
    it = sequences_.emplace_hint(it, id, std::move(sequence));
    return &it->second;
}

/**
 * \brief 根据序列的id，从序列缓存中获取序列
 * \param [in] id 序列的id
 * \return
 * \note 当request的start_flag == false的时候，调用 Get。如果在 sequence_ 中命中，那么就直接返回命中的序列。否则返回 nullptr
*/
const Sequence* SequenceManager::Get(uint64_t id)
{
    if (auto it = sequences_.find(id); it != sequences_.end()) {
        return &it->second;
    }
    return nullptr;
}

bool SequenceManager::Contains(uint64_t id)
{
    return sequences_.find(id) != sequences_.end();
}

/**
 * \param [in, out] it 要删除的序列在序列缓存中的位置
 * \note 什么时候seq会被erase？request中有sequence_end时。
*/
void SequenceManager::Erase(std::map<uint64_t, Sequence>::iterator& it)
{
    auto& seq = it->second;
    if (seq.status == Sequence::kCached) {
        // 获取seq中valid的block数量
        const int count = block_manager_->Verify(seq.blocks, seq.block_unique_ids);
        seq.blocks.resize(count);
    }
    else {
        UpdateAndSetUnlock(seq);
    }
    freed_.insert(freed_.end(), seq.blocks.begin(), seq.blocks.end());
    it = sequences_.erase(it);
}

bool SequenceManager::Erase(uint64_t id)
{
    if (auto it = sequences_.find(id); it != sequences_.end()) {
        Erase(it);
        return true;
    }
    return false;
}

/**
 * \brief 为 sequences 锁定被cache的block，避免其他sequence抢占或者强制换出
*/
void SequenceManager::VerifyAndLockCached(const Sequences& sequences)
{
    BlockIds blocks;
    for (const auto& p : sequences) {
        auto& seq = const_cast<Sequence&>(*p);
        if (seq.status != Sequence::kCached) {
            continue;
        }
        FT_CHECK(seq.blocks.size() == seq.block_unique_ids.size());
        // Verify cache blocks that may be invalidated
        const int count = block_manager_->Verify(seq.blocks, seq.block_unique_ids);
        seq.blocks.resize(count);
        seq.block_unique_ids.resize(count);

        blocks.insert(blocks.end(), seq.blocks.begin(), seq.blocks.end());
        seq.cache_len = std::min<int>(seq.cache_len, seq.blocks.size() * block_seq_len_);
        seq.status    = Sequence::kLocked;
    }
    block_manager_->Lock(blocks);
}

/**
 * 提交 unlock 和 free 的 block
 * 被 unlock 的 block 一定都是 active 的block。`UpdateAndSetUnlock`会把 active/locked 的 sequence的 block 加入到unlocked_里面
 * 被free的block一定是 cached 的 block
*/
void SequenceManager::CommitUnlockAndFree()
{
    if (!unlocked_.empty()) {
        block_manager_->Unlock(unlocked_);
        unlocked_.clear();
    }

    if (!freed_.empty()) {
        block_manager_->Free(freed_);
        freed_.clear();
    }
}

/**
 * \brief 更新非 cached 状态的sequence的资源
 * sequence 的所有 blocks（这些block必须都是active状态）的 timestamp 增加。timestamp和block换出策略有关（LRU）
 * 把sequence的block都插入到unlocked_ blocks中
 * 序列状态改成 cached
*/
void SequenceManager::UpdateAndSetUnlock(const Sequence& sequence)
{
    FT_CHECK(sequence.status != Sequence::kCached);
    auto& seq = const_cast<Sequence&>(sequence);
    block_manager_->Touch(seq.blocks);
    unlocked_.insert(unlocked_.end(), seq.blocks.begin(), seq.blocks.end());
    seq.status = Sequence::kCached;
}

namespace {

struct Schedule {
    int free;   // free block 数量（没有sequence在用）
    int cached; // cache block 数量

    int allocate{}; //需要申请的block的数量
    int evict{};    //需要换出去的block的数量
    int preempt{};  //需要抢占的block的数量

    int last;   // 初始为 #sequences (当前的 + incoming的)

    int input_count1;   // 暂时理解为 max_context_token_num
    int input_count2;   // 暂时理解为 max_context_token_num

    Sequences        active;
    std::vector<int> block_counts;
    Sequences        inactive;
    Sequences        victims;

    /**
     * \param [in] snapshot 当前 block mgr中的快照
     * \param [in] size 序列的数量: 当前在处理的 + incoming 的
     * \param [in] _input_count1
     * \param [in] _input_count1
    */
    Schedule(Snapshot snapshot, int size, int _input_count1, int _input_count2):
        free(snapshot.free),
        cached(snapshot.cached),
        last(size),
        use_count_(std::move(snapshot.use_count)),
        unlocked_(size),
        it_(size),
        input_count1(_input_count1),
        input_count2(_input_count2)
    {
    }

    /**
     * 为什么不直接看seqs[vidx]占用的block的使用情况，而是要进行一个循环呢？++lvhan (23.12.27)
    */
    int Unlock(const Sequences& seqs, int vidx)
    {
        while (vidx < it_) {
            const auto& blocks = seqs[--it_]->blocks;
            int         count  = 0;
            for (const auto& bid : blocks) {
                count += static_cast<int>(--use_count_[bid] == 0);
            }
            // 编号为 it_ 的sequence有多少在用的block（有可能它刚开始申请的block，还没有被用，毕竟是一步步的forward的
            unlocked_[it_] = count;
        }
        return unlocked_[vidx];
    }

private:
    std::vector<int> use_count_; // blockmgr中，所有 block 被 active sequences 使用的次数
    std::vector<int> unlocked_; // 初始长度 #sequences (当前的 + incoming的)
    int              it_; // 初始为 #sequences (当前的 + incoming的)
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i) {
        os << (i ? "," : "") << v[i];
    }
    os << "]";
    return os;
}

std::ostream& operator<<(std::ostream& os, const Schedule& s)
{
    os << "free=" << s.free << ", cached=" << s.cached << ", allocate=" << s.allocate << ", evict=" << s.evict
       << ", preempt=" << s.preempt << ", active=" << s.active << ", victims=" << s.victims
       << ", block_counts=" << s.block_counts << ", inactive=" << s.inactive;
    return os;
}

struct Transaction {
    int index_;         // transaction对应的sequence在sequences中的编号
    int block_count_;   // transaction对应的sequence所需要的block数量
    int input_count_;   // sequences_[index_] 当前的 input_length

    int allocate_{};    // 从free blocks中申请的数量
    int evict_{};       // 从cache blocks中驱逐走的数量
    int preempt_{};     // 抢占的数量

    Sequences victims_;

    const Sequences& sequences_;
    Schedule&        schedule_;

    /**
     * \brief
     * \param [in] sequences 在处理的序列的集合（当前的 + incoming的）
     * \param [in] index 一个序列在 sequences 中的下标
     * \param [in] block_count sequences[index] 需要的 block 的数量
     * \param [in] input_count sequences[index] 当前的input_length。对于incoming的sequence来说，它是input_ids的length。对于在处理的sequence来说，它的值是1（新生成的token）
     * \param [in, out] sched 调度器
     * \note
    */
    explicit Transaction(const Sequences& sequences, int index, int block_count, int input_count, Schedule& sched):
        sequences_(sequences), schedule_(sched), index_(index), block_count_(block_count), input_count_(input_count)
    {
    }

    /**
     * \brief
    */
    void Process()
    {
        // ++ lvhan (23.12.26) 这个分支是不是总为真？什么情况下为假？Commit函数会更新参数
        if (schedule_.input_count1 > 0) {
            int count = block_count_;

            // 先从 free blocks 中申请
            int tmp = std::min(schedule_.free, count);
            count -= tmp;
            allocate_ += tmp;
            // 如果不够，就再从 cached 中申请
            tmp = std::min(schedule_.cached, count);
            count -= tmp;
            evict_ += tmp;

            // 当前在处理的序列的编号是index_, 从最后一个序列开始往前，最后的序列优先级最低。
            for (int vidx = schedule_.last - 1; count && vidx > index_; --vidx) {
                // ++ lvhan(23.12.26) 为啥忽略 cached sequence呢？因为incoming的sequence都是kCached的，是新的序列，需要尽快处理这些序列，给用户响应
                if (sequences_[vidx]->status == Sequence::kCached) {
                    continue;
                }
                // locked & active 的sequence 成了受害者，也就是当前在处理的sequence
                victims_.push_back(sequences_[vidx]);
                preempt_ += schedule_.Unlock(sequences_, vidx);

                if (count <= preempt_) {
                    evict_ += count;
                    count -= count;
                    schedule_.last = vidx;  // ! modifiying `sched_.last` is part of commit
                    break;
                }
            }
            if (count == 0) { // block 资源能满足，提交transaction
                return Commit();
            }
        }
        // ++ lvhan (23.12.26) 为啥要改为0，并且放在 inactive中 ？
        // 因为前面没有走到 Commit，表示这个序列获取不到要求的block资源，所以就变成 inactive
        // input_length为0，表示之后要重新 prefill
        const_cast<Sequence*>(sequences_[index_])->input_length = 0;
        schedule_.inactive.push_back(sequences_[index_]);
    }

    void Commit()
    {
        // update available resources
        schedule_.free -= allocate_;
        FT_CHECK(schedule_.free >= 0);
        schedule_.cached += preempt_;
        schedule_.cached -= evict_;
        FT_CHECK(schedule_.cached >= 0);

        // update scheduled operations
        schedule_.allocate += allocate_;
        schedule_.evict += evict_;
        schedule_.preempt += preempt_;
        schedule_.victims.insert(schedule_.victims.end(), victims_.begin(), victims_.end());

        // update active sequences
        schedule_.active.push_back(sequences_[index_]);
        schedule_.block_counts.push_back(block_count_);

        if (input_count_ > schedule_.input_count2) {
            input_count_ = schedule_.input_count1;
        }
        schedule_.input_count1 -= input_count_;
        schedule_.input_count2 -= input_count_;
        const_cast<Sequence*>(sequences_[index_])->input_length = input_count_;
    }
};

std::ostream& operator<<(std::ostream& os, const Transaction& trans)
{
    os << "index=" << trans.index_ << ", block_count=" << trans.block_count_ << ", allocate=" << trans.allocate_
       << ", evict=" << trans.evict_ << ", preempt=" << trans.preempt_ << ", victims=" << trans.victims_;
    return os;
}

}  // namespace

/**
 * \brief 把 sequences 和 context_lengths 按照 priorities 从小到大排序。结合 llamabatch中对优先级的规定,
 * 也就是先来先服务原则
*/
void SequenceManager::SortByPriority(Sequences&                   sequences,
                                     std::vector<int>&            context_lengths,
                                     const std::vector<uint64_t>& priorities)
{
    // sort according to priority
    std::vector<int> idxs(sequences.size());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(), [&](int i, int j) {
        return priorities[i] < priorities[j];  //
    });
    Sequences        tmp_sequences(sequences.size());
    std::vector<int> tmp_lengths(context_lengths.size());
    for (int i = 0; i < sequences.size(); ++i) {
        tmp_sequences[i] = sequences[idxs[i]];
        tmp_lengths[i]   = context_lengths[idxs[i]];
    }
    sequences.swap(tmp_sequences);
    context_lengths.swap(tmp_lengths);
}

// template<class P, class... Ts>
// void SortByPriority(const std::vector<P>& priorities, Ts&... ranges)
// {
//     // sort according to priority
//     std::vector<int> idxs(priorities.size());
//     std::iota(idxs.begin(), idxs.end(), 0);
//     std::sort(idxs.begin(), idxs.end(), [&](int i, int j) {
//         return priorities[i] < priorities[j];  //
//     });
//     auto reorder = [&](auto& src) {
//         auto dst = src;
//         for (size_t i = 0; i < idxs.size(); ++i) {
//             dst[i] = src[idxs[i]];
//         }
//         src.swap(dst);
//     };
//     (reorder(ranges), ...);
// }

/** \brief 返回每个sequence所需要的block
 *
*/
std::vector<int> SequenceManager::CountRequiredBlocks(const Sequences&        sequences,
                                                      const std::vector<int>& context_lengths,
                                                      int                     step_length)
{
    std::vector<int> required(sequences.size());
    for (int i = 0; i < sequences.size(); ++i) {
        int seq_len = context_lengths[i] + step_length;
        int count   = (seq_len + block_seq_len_ - 1) / block_seq_len_ - static_cast<int>(sequences[i]->blocks.size());
        required[i] = std::max(0, count);
    }
    return required;
}

/**
 * \param [in] sequences active sequence的列表
 * \param [in] counts 存每个active sequence当前所用的block数量。它的长度和`sequences`的长度相同
 * \param [in] blocks 申请到的block
 * \param [in] unique_ids。申请到的block的unique_id的集合。它的长度和blocks的长度相同
*/
void SequenceManager::AssignAndActivate(const Sequences&        sequences,  //
                                        const std::vector<int>& counts,
                                        const BlockIds&         blocks,
                                        const UniqueIds&        unique_ids)
{
    FT_CHECK(sequences.size() == counts.size());
    int first = 0;
    for (int i = 0; i < sequences.size(); ++i) {
        auto& s     = const_cast<Sequence&>(*sequences[i]);
        auto  count = counts[i];
        int   last  = first + count;
        FT_CHECK(last <= blocks.size());
        s.blocks.insert(s.blocks.end(), blocks.begin() + first, blocks.begin() + last);
        s.block_unique_ids.insert(s.block_unique_ids.end(), unique_ids.begin() + first, unique_ids.begin() + last);
        s.status = Sequence::kActive;
        first    = last;
    }
}

/**
 * \param [in] sequences 当前在处理的sequence，以及新加入的Sequence
 * \param [in] context_lengths `sequences` 中每个序列的 context_length
 * \param [in] priorities   `sequences` 中每个序列的优先级
 * \param [in] step_length decoding阶段的步长
*/
auto SequenceManager::Materialize(Sequences                    sequences,
                                  std::vector<int>             context_lengths,
                                  const std::vector<uint64_t>& priorities,
                                  int                          step_length,
                                  AdjustInputCount             adjust) -> Outcome
{
    ////////////////////////////////////////////////////////////////////////////////
    /// Schedule the assignment of blocks to sequences

    // process deferred unlock and free operations
    CommitUnlockAndFree();

    // 根据先来先服务的顺序，对sequences进行排序
    SortByPriority(sequences, context_lengths, priorities);

    // SortByPriority(priorities, sequences, context_lengths);

    // Verify and lock cache sequences to avoid their blocks being evicted unnoticed
    // the blocks can still be preempted later
    // Materialize 被 llamabatch::ProcessInferRequests调用，对于新序列（request.start_flag==true，它是 kCached的状态
    VerifyAndLockCached(sequences);

    // ++ lvhan（23.12.26）这里是用Dynamic SplitFuse。先略过。input_count1, input_count2 都是 max_context_token_num
    auto [input_count1, input_count2] = adjust(sequences, context_lengths);

    std::vector<int> required = CountRequiredBlocks(sequences, context_lengths, step_length);
    // dbg(required);

    Schedule schedule(block_manager_->TakeSnapshot(), sequences.size(), input_count1, input_count2);

    // `schedule.last` is decreasing in the loop
    // 从高优先级的sequence开始，分配它所需要的 block。首先会从free blocks中申请，如果不够，就从cache block中申请。
    // free block、cache block数量都存在了schedule中，通过blockmgr的快照拿到。
    // sequence 所需要的数量在前面的 CountRequiredBlocks中算出来了
    for (int i = 0; i < schedule.last; ++i) {
        const int input_length = context_lengths[i] - sequences[i]->cache_len;
        Transaction{sequences, i, required[i], input_length, schedule}.Process();
    }

    // mark remaining sequences invalid
    // 上一个步骤 Transaction.Process，会从schedule.last倒序访问。高优先级的序列可能会抢占
    // 低优先级序列的block，被抢占的sequence，会加入到inactive数组中
    for (int i = schedule.last; i < sequences.size(); ++i) {
        schedule.inactive.push_back(sequences[i]);
    }

    ////////////////////////////////////////////////////////////////////////////////
    /// Schedule is ready, time to execute it. (locked -> cached -> free -> locked)

    // combine allocate and evict since evicted blocks are reused by allocation
    schedule.allocate += schedule.evict;

    if (schedule.allocate) {
        dbg(*block_manager_);
    }

    Outcome outcome{};
    outcome.allocation = schedule.allocate;
    outcome.swap_in    = std::count_if(schedule.active.begin(), schedule.active.end(), [](auto p) {
        if (p->status != Sequence::kActive) {
            dbg(*p);
        }
        return p->status != Sequence::kActive;  //
    });
    outcome.swap_out   = std::count_if(schedule.inactive.begin(), schedule.inactive.end(), [](auto p) {
        if (p->status == Sequence::kActive) {
            dbg(*p);
        }
        return p->status == Sequence::kActive;  //
    });

    // release preempted blocks -> cached
    if (!schedule.victims.empty()) {
        for (const auto& p : schedule.victims) {
            UpdateAndSetUnlock(*p);
        }
        CommitUnlockAndFree();
    }

    // evict cached blocks -> free
    if (schedule.evict) {
        block_manager_->Evict(schedule.evict);
    }

    // allocate & assign blocks
    {
        BlockIds  block_ids;
        UniqueIds unique_ids;
        if (schedule.allocate) {
            std::tie(block_ids, unique_ids) = block_manager_->Allocate(schedule.allocate);
        }
        AssignAndActivate(schedule.active, schedule.block_counts, block_ids, unique_ids);
    }

    // active -> locked
    for (const auto& p : schedule.inactive) {
        if (p->status == Sequence::kActive) {
            const_cast<Sequence*>(p)->status = Sequence::kLocked;
        }
    }

    // TM_LOG_ERROR("active: %4d, cached: %4d, free: %4d",
    //              block_manager_->active_count(),
    //              block_manager_->cached_count(),
    //              block_manager_->free_count());

    return outcome;
}

}  // namespace turbomind
