#pragma once
#include "MultiThread/AutoResetEvent.hpp"
#include <boost/intrusive_ptr.hpp>
#include <boost/smart_ptr/intrusive_ptr.hpp>
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

public:
  enum AsyncState { WAITING_EXECUTION, RESOLVED, FAILED };

protected:
  enum AsyncType { START, THEN, RESULT };
  AsyncState _state = WAITING_EXECUTION;
  DynData _result;
  boost::intrusive_ptr<Task> _father{};
  boost::intrusive_ptr<Task> _children{};
  boost::intrusive_ptr<Task> _failureHandler{};

  std::optional<std::function<void(Task *)>> onCompleted{};
  void resolve(bool failed = false) {
    _state = failed ? FAILED : RESOLVED;
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
  void set_children(boost::intrusive_ptr<Task> task) { _children = task; }
  void set_father(boost::intrusive_ptr<Task> task) { _father = task; }
  void set_failure(boost::intrusive_ptr<Task> task) { _failureHandler = task; }
  void set_resolve_handler(std::function<void(Task *)> fnc) {
    onCompleted.emplace(fnc);
  }
  bool child_of(Task *t);
  bool father_of(Task *t);
  void cancel();
};

class PostableTask : public Task {
private:
  std::function<void()> _fnc;

public:
  PostableTask(std::function<void()> fnc) : _fnc(fnc) {}
  virtual void operator()() {
    _fnc();
    resolve();
  }
};

class Executor {
public:
  enum State { IDLE, PROCESSING, WAITING_EXECUTION };
#ifndef UNITTEST
protected:
#endif
  std::thread *_executingThread = nullptr;
  std::queue<boost::intrusive_ptr<Task>> mq{};
  State status = IDLE;
  bool _end = false;
  std::mutex queueLock{};
  std::optional<boost::intrusive_ptr<Task>> take();
  void push(boost::intrusive_ptr<Task> task);

public:
  Executor();
  void assign(boost::intrusive_ptr<Task>);
  boost::intrusive_ptr<Task> post(std::function<void()> f);
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
  boost::intrusive_ptr<Task> post(std::function<void()> f);
  void schedule(boost::intrusive_ptr<Task> task);
  void start();
  void stop();
  void reset();

  void setMaxEnqueueDegree(int maxdegree) { _maxEnqueueDegree = maxdegree; }
  static Scheduler &main();
  bool AssignToIdle(boost::intrusive_ptr<Task> task);
  bool AssignToLowerCharged(boost::intrusive_ptr<Task> task);
  std::optional<boost::intrusive_ptr<Task>> take();
#ifndef UNITTEST
protected:
#endif
  std::queue<boost::intrusive_ptr<Task>> mq{};
  Scheduler();
  ~Scheduler();
  static Scheduler *_instance;

  void routine();
};

void intrusive_ptr_add_ref(Task *p);

void intrusive_ptr_release(Task *p);