#pragma once
#include "MultiThread/AutoResetEvent.hpp"
#include <condition_variable>
#include <cstring>
#include <memory>
#include <queue>
#include <thread>
#include <vector>

class DynData {
private:
  void *ptr = nullptr;
  size_t size;

public:
  template <typename T> void emplace(T value) {
    ptr = malloc(sizeof(T));
    std::memcpy(ptr, &value, sizeof(T));
    size = sizeof(T);
  }

  template <typename T> T &reintepret() {
    return reinterpret_cast<T &>(*((T *)ptr));
  }

  bool has_value() const { return ptr != nullptr; }

  ~DynData() {
    if (ptr != nullptr) {
      free(ptr);
    }
  }
};

class Executor;
class Task {
  friend class Executor;

protected:
  enum AsyncType { START, THEN, RESULT };
  enum AsyncState { WAITING_EXECUTION, RESOLVED };
  AsyncState _state = WAITING_EXECUTION;
  DynData _result;
  ManualResetEvent _executed{false};
  Task *_father = nullptr;
  Task *_children = nullptr;

public:
  virtual void operator()() = 0;
  virtual ~Task() = default;
  AsyncState state() const { return _state; }
  DynData &result() { return this->_result; }
  void wait() { _executed.WaitOne(); }
  void set_children(Task *task) { _children = task; }
  void set_father(Task *task) { _father = task; }
};

class Executor {
  enum State { IDLE, PROCESSING, WAITING_EXECUTION };
#ifndef UNITTEST
protected:
#endif
  std::thread _executingThread{};
  std::queue<Task *> mq{};
  State status = IDLE;
  bool _end = false;

public:
  Executor();
  void assign(Task *task);
  ~Executor();
};

class Scheduler {
private:
  std::vector<Executor> _executors{};
  int _previousCount = 0;

public:
  void schedule(Task *task);
  void start();
  void stop();
  void reset();
  static Scheduler &main();
#ifndef UNITTEST
protected:
#endif
  std::queue<Task *> mq{};
  Scheduler() {}
  static Scheduler *_instance;
};
