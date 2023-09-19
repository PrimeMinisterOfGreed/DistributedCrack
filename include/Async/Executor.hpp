#pragma once
#include "MultiThread/AutoResetEvent.hpp"
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <memory>
#include <optional>
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
      ptr = nullptr;
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
public:
  enum State { IDLE, PROCESSING, WAITING_EXECUTION };
#ifndef UNITTEST
protected:
#endif
  std::thread *_executingThread = nullptr;
  std::queue<Task *> mq{};
  State status = IDLE;
  bool _end = false;

public:
  Executor();
  void assign(Task *task);
  int count() const { return mq.size(); }
  State state() const { return status; }
  ~Executor();
};

class Scheduler {
private:
  std::vector<Executor *> _executors{};
  int _previousCount = 0;
  bool _end = false;
  std::thread *_executionThread = nullptr;

public:
  void schedule(Task *task);
  void start();
  void stop();
  void reset();

  static Scheduler &main();
  bool AssignToIdle(Task *task);
  bool AssignToLowerCharged(Task *task);
#ifndef UNITTEST
protected:
#endif
  std::queue<Task *> mq{};
  Scheduler();
  ~Scheduler();
  static Scheduler *_instance;

  void routine();
};
