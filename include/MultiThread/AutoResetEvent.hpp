#pragma once
#include <future>
#include <mutex>
#include <condition_variable>

class ResettableEvent
{
  private:
    std::mutex _lock;
    std::condition_variable _var;
    bool _flag;
    bool _autoreset = false;
  public:
    void Set();
    void Reset();
    virtual void WaitOne();
    ResettableEvent(bool Autoreset, bool initialState);
};

class ManualResetEvent : public ResettableEvent
{
  public:
    ManualResetEvent(bool initialState): ResettableEvent(false,initialState){}
};

class AutoResetEvent : public ResettableEvent
{
  public:
    AutoResetEvent(bool initialState) : ResettableEvent(true,initialState){}
};