#include "MultiThread/AutoResetEvent.hpp"
#include <mutex>

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
