// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/utils/Tensor.h"
#include <condition_variable>
#include <cstdint>
#include <future>
#include <limits>
#include <queue>
#include <unordered_map>

namespace turbomind {

struct Request {
    uint64_t id;        // sequence id
    uint64_t unique_id; // monotonic increasing

    bool start_flag;    // 是否序列的第一个请求
    bool end_flag;      // 是否结束整个序列
    bool stop_flag;     // 是否停止生成

    // per rank inputs/outputs
    std::vector<TensorMap> inputs;
    std::vector<TensorMap> outputs;

    using Callback = std::function<void(std::unordered_map<std::string, Tensor>*)>;
    Callback stream_cb;

    enum
    {
        kInvalid  = 1,
        kConflict = 2,
        kBusy     = 3,
        kInactive = 4,
        kFail     = 5,
        kTooLong  = 6
    };
    std::promise<int> signal;
};

class RequestQueue {
public:
    /**
     * \brief LlamaV2<T>::forward 在接到请求后，会调用这个函数，把请求入队。
     * 标记为 stop 的request 入队 stop_queue_，其余的入队 infer_queue_
     * 只有 rank 0 的 instance 执行入队操作。enqueue为每个request返回一个future。
     * caller 拿到 future 之后，调用 future.get()，就处于一种阻塞的状态
    */
    std::vector<std::future<int>> enqueue(std::vector<std::shared_ptr<Request>> requests)
    {
        std::vector<std::future<int>> futures;
        futures.reserve(requests.size());
        {
            std::lock_guard<std::mutex> lock(mutex_);

            if (closed_) {
                throw std::runtime_error("Queue is closed");
            }

            for (auto& r : requests) {
                futures.push_back(r->signal.get_future());
                if (r->stop_flag) {
                    stop_queue_.push(std::move(r));
                }
                else {
                    infer_queue_.push(std::move(r));
                }
            }
        }
        cv_.notify_one();
        return futures;
    }

    /**
     * \brief 从 stop_queue_ 和 infer_queue_ 中取请求
     * \param [in, out] stop_requests 标记为 stop 的请求
     * \param [in, out] infer_requests 推理请求
     * \param [in] max_infer_count 最多取推理请求的个数，和batch的空闲slot数一致
     * \param [in] blocking 是否阻塞。当batch的空闲slot数为0时，阻塞式，其他非阻塞式
     * \param [in, out] abort caller是不是发了abort
    */
    void dequeue(std::vector<std::shared_ptr<Request>>& stop_requests,
                 std::vector<std::shared_ptr<Request>>& infer_requests,
                 unsigned                               max_infer_count,
                 bool                                   blocking,
                 bool&                                  abort)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (blocking) {
            // 等待直到 stop_queue_ 或者 infer_queue_ 中有数据
            cv_.wait(lock, [this] { return !(stop_queue_.empty() && infer_queue_.empty()) || closed_; });
            if (closed_) {
                abort = true;
                return;
            }
        }

        // stop_queue_中的请求全部取出来
        stop_requests.clear();
        while (!stop_queue_.empty()) {
            stop_requests.push_back(std::move(stop_queue_.front()));
            stop_queue_.pop();
        }

        // infer_queue_ 中的请求至多取 `max_infer_count` 个
        infer_requests.clear();
        while (!infer_queue_.empty() && infer_requests.size() < max_infer_count) {
            infer_requests.push_back(std::move(infer_queue_.front()));
            infer_queue_.pop();
        }
    }

    void close()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        cv_.notify_all();
    }

private:
    std::queue<std::shared_ptr<Request>> stop_queue_;
    std::queue<std::shared_ptr<Request>> infer_queue_;
    std::mutex                           mutex_;
    std::condition_variable              cv_;
    bool                                 closed_{false};
};

}  // namespace turbomind
