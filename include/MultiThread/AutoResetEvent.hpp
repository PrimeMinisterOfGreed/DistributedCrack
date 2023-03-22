#pragma once
#include "Concepts.hpp"
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <vector>

class WaitHandle
{
    friend int WaitAny(std::vector<WaitHandle *> &handles);

  protected:
    virtual void SetWakeupFunction(std::function<void()> function) = 0;
    virtual void UnsetWakeupFunction() = 0;

  public:
    virtual void WaitOne() = 0;
};
class ResettableEvent : public WaitHandle
{

  private:
    std::mutex _lock;
    std::condition_variable _var;
    bool _flag;
    bool _autoreset = false;
    std::function<void()> *_wakeUpFunction = nullptr;

  protected:
    void SetWakeupFunction(std::function<void()> function) final;
    void UnsetWakeupFunction() final;

  public:
    void Set();
    void Reset();
    virtual void WaitOne();
    ResettableEvent(bool Autoreset, bool initialState);
};

class ManualResetEvent : public ResettableEvent
{
  public:
    ManualResetEvent(bool initialState) : ResettableEvent(false, initialState)
    {
    }
};

class AutoResetEvent : public ResettableEvent
{
  public:
    AutoResetEvent(bool initialState) : ResettableEvent(true, initialState)
    {
    }
};

int WaitAny(std::vector<WaitHandle *> &handles);