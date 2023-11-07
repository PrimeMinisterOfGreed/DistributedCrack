#pragma once
#include "Async.hpp"
#include "Async/Executor.hpp"
#include "Concepts.hpp"
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>

template <typename... Args> struct AsyncLoop : public Future<void, Args...> {
private:
  std::tuple<Args...> _args;

public:
  template <typename IterF, typename Predicate>
  AsyncLoop(IterF &&fnc, Predicate &&terminator) {}

  virtual void operator()() override {}
};

template <typename... Args> struct ParallelLoop : public Task {

protected:
  std::tuple<Args...> _args;
  std::vector<sptr<Task>> _tasks;
  size_t _opCounter = 0;
  std::vector<std::function<void(Args...)>> _fncs;
  std::mutex _queueLock{};
  void set_args(Args... args) { _args = std::tuple<Args...>{args...}; }
  int iterations = 1;
  template <typename... Fs>
  ParallelLoop(int it, Fs... f)
      : _fncs(std::vector<std::function<void(Args...)>>{f...}), iterations(it) {

  }

public:
  virtual void operator()() {
    for (int i = 0; i < iterations; i++) {
      for (auto f : _fncs) {
        std::apply(
            [this, f](Args... args) {
              auto t = Future<void, Args...>::Create(f, args...);
              *t += [this]() {
                std::lock_guard _{_queueLock};
                _opCounter--;
                if (_opCounter == 0) {
                  this->resolve();
                }
              };
              _tasks.push_back(t);
            },
            _args);
      }
    }
  }

  template <typename... Fs>
  static sptr<ParallelLoop<Args...>> Create(int it, Fs... f) {
    return sptr<ParallelLoop<Args...>>(new ParallelLoop<Args...>(it, f...));
  }
  template <typename... Fs> static auto Create(Fs... f) {
    return Create(0, f...);
  }

  void Run(Args... args) {
    _args = std::tuple{args...};
    Scheduler::main().schedule(sptr<Task>{this});
  }
};
