#include "MultiThread/AutoResetEvent.hpp"
#include <mutex>
#include <vector>

void ResettableEvent::SetWakeupFunction(std::function<void()> function)
{
    _wakeUpFunction = &function;
}

void ResettableEvent::UnsetWakeupFunction()
{
    _wakeUpFunction = nullptr;
}

void ResettableEvent::Set()
{
    std::lock_guard<std::mutex> l(_lock);
    _flag = true;
    _var.notify_one();
    if (_wakeUpFunction != nullptr)
    {
        auto fnc = *_wakeUpFunction;
        fnc();
    }
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

int WaitAny(std::vector<WaitHandle *> &handles)
{
    AutoResetEvent wakeup{false};
    std::unique_lock<std::mutex> lock{};
    int result = -1;
    for (int i = 0; i < handles.size(); i++)
    {
        handles[i]->SetWakeupFunction([&result, i, &wakeup, &lock]() {
            lock.lock();
            result = i;
            wakeup.Set();
        });
    }
    wakeup.WaitOne();
    int constResult = result;
    lock.unlock();
    return constResult;
};
