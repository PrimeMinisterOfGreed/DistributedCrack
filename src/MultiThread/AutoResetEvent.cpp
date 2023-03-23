#include "MultiThread/AutoResetEvent.hpp"
#include <mutex>
#include <thread>
#include <vector>

void ResettableEvent::Set()
{
    std::lock_guard<std::mutex> l(_lock);
    _flag = true;
    _var.notify_one();
}

void ResettableEvent::Reset()
{
    std::lock_guard<std::mutex> l(_lock);
    _flag = false;
}

void ResettableEvent::WaitOne()
{
    std::unique_lock<std::mutex> l(_lock);
    while (!_flag)
    {
        _var.wait(l);
    }
    if (_autoreset)
        _flag = false;
}

ResettableEvent::ResettableEvent(bool Autoreset, bool initialState)
{
    _autoreset = Autoreset;
    _flag = initialState;
}

int WaitAny(std::vector<WaitHandle *> handles)
{
    WaitHandle *result = nullptr;
    std::mutex lock{};
    AutoResetEvent onResult{false};
    for (auto &handle : handles)
    {
        auto t = std::thread([&handle, &result, &onResult, &lock] {
            handle->WaitOne();
            lock.lock();
            if (result == nullptr)
            {
                result = handle;
                onResult.Set();
            }
        });
        t.detach();
    }
    int index = 0;
    onResult.WaitOne();
    for (int i = 0; i < handles.size(); i++)
    {
        if (result == handles[i])
        {
            index = i;
            break;
        }
    }

    lock.unlock();
    for (auto &handle : handles)
        handle->Set();
    for (auto &handle : handles)
        handle->Reset();
    return index;
};
