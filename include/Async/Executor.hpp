#pragma once
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

class Task {
protected:
  DynData _result;

public:
  virtual void operator()() = 0;
  virtual ~Task() = default;
  DynData &result() { return this->_result; }
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

public:
  void schedule(Task *task);
  void start();
  void stop();
  static Scheduler &main();
#ifndef UNITTEST
protected:
#endif
  std::queue<Task *> mq{};
  Scheduler() {}
  static Scheduler *_instance;
};
