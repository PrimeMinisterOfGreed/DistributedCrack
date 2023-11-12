#pragma once
#include "Async.hpp"
#include "Async/Executor.hpp"
#include "Concepts.hpp"
#include "Functions.hpp"
#include <algorithm>
#include <boost/graph/properties.hpp>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>

template <typename... Args> struct AsyncLoop : public Task {

  template <typename T> struct LoopRequest {
    bool end = false;
    T hold;
  };

private:
  std::tuple<Args...> _args;
  sptr<Task> _innerTask;

public:
  template <typename IterF> AsyncLoop(IterF &&fnc) {}

  virtual void operator()(sptr<Task> thisptr) override {}
};

template <typename... Args> struct ParallelLoop : public Task {

protected:
  std::tuple<Args...> _args;
  size_t _opCounter = 0;
  std::vector<std::function<void(Args...)>> _fncs;
  std::mutex _queueLock{};
  void set_args(Args... args) { _args = std::tuple<Args...>{args...}; }
  int iterations = 1;
  template <typename... Fs>
  ParallelLoop(int it, Fs... f)
      : Task(), _fncs(std::vector<std::function<void(Args...)>>{f...}),
        iterations(it) {}

public:
  virtual void operator()(sptr<Task> thisptr) {
    std::vector<sptr<Task>> tasks{};
    if (iterations <= 0) {
      resolve();
      return;
    }
    for (auto f : _fncs) {
      std::apply(
          [this, f, &tasks, thisptr](Args... args) {
            auto t = Future<void, Args...>::Create(f, args...);
            *t += [this, t, thisptr]() {
              std::lock_guard _{_queueLock};
              _opCounter--;
              if (_opCounter == 0) {
                iterations--;
                Scheduler::main().schedule(thisptr);
              }
            };
            tasks.push_back(t);
          },
          _args);
    }
    for (auto t : tasks) {
      Scheduler::main().schedule(t);
    }
    _opCounter = tasks.size();
  }

  template <typename... Fs>
  static sptr<ParallelLoop<Args...>> Run(int it, Args... args, Fs... f) {
    auto ptr = sptr<ParallelLoop<Args...>>(new ParallelLoop<Args...>(it, f...));
    ptr->set_args(args...);
    Scheduler::main().schedule(ptr);
    return ptr;
  }
  template <typename... Fs> static auto Run(Fs... f) { return Run(1, f...); }

  ~ParallelLoop() { wait(); }
};

template <typename... Args> class AsyncSILoop : public Task {
  std::tuple<Args...> _args;
  std::vector<sptr<Task>> _tasks;
  std::mutex _lock{};
  std::function<void(size_t, Args...)> _iterFnc;
  size_t _iterations = 1;

public:
  template <typename F>
  AsyncSILoop(F &&fnc, size_t iterations, Args... args)
      : _args(args...), _iterFnc(fnc), _iterations(iterations) {}

  virtual void operator()(sptr<Task> thisptr) {
    for (size_t i = 0; i < _iterations; i++) {
      std::apply(
          [this, i](Args... args) {
            auto t =
                Future<void, size_t, Args...>::Create(_iterFnc, i, args...);
            *t += [this]() {
              std::lock_guard l(_lock);
              _iterations--;
              if (_iterations == 0) {
                resolve();
              }
            };
            _tasks.push_back(t);
          },
          _args);
    }
    for (auto t : _tasks) {
      Scheduler::main().schedule(t);
    }
  }

  template <typename F>
  static sptr<AsyncSILoop<Args...>> Run(size_t iterations, F &&fnc,
                                        Args... args) {
    auto t = sptr<AsyncSILoop<Args...>>(
        new AsyncSILoop<Args...>(fnc, iterations, args...));
    Scheduler::main().schedule(t);
    return t;
  }
};