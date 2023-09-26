#pragma once
#include "MultiThread/AutoResetEvent.hpp"
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
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
  ManualResetEvent _executed{false};

protected:
  enum AsyncType { START, THEN, RESULT };
  enum AsyncState { WAITING_EXECUTION, RESOLVED };
  AsyncState _state = WAITING_EXECUTION;
  DynData _result;
  Task *_father = nullptr;
  Task *_children = nullptr;
  std::optional<std::function<void(Task *)>> onCompleted{};
  void resolve() {
    _state = RESOLVED;
    _executed.Set();
    if (onCompleted.has_value()) {
      onCompleted.value()(this);
      onCompleted.reset();
    }
  }

public:
  virtual void operator()() = 0;
  virtual ~Task() = default;
  AsyncState state() const { return _state; }
  DynData &result() { return this->_result; }
  void wait() { _executed.WaitOne(); }
  void set_children(Task *task) { _children = task; }
  void set_father(Task *task) { _father = task; }
  void set_resolve_handler(std::function<void(Task *)> fnc) {
    onCompleted.emplace(fnc);
  }
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
  std::mutex queueLock{};
  std::optional<Task *> take();
  void push(Task *task);

public:
  Executor();
  void assign(Task *task);
  void start();
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
  std::mutex schedLock{};
  int _maxEnqueueDegree = 10;

public:
  void schedule(Task *task);
  void start();
  void stop();
  void reset();
  void setMaxEnqueueDegree(int maxdegree) { _maxEnqueueDegree = maxdegree; }
  static Scheduler &main();
  bool AssignToIdle(Task *task);
  bool AssignToLowerCharged(Task *task);
  std::optional<Task *> take();
#ifndef UNITTEST
protected:
#endif
  std::queue<Task *> mq{};
  Scheduler();
  ~Scheduler();
  static Scheduler *_instance;

  void routine();
};
