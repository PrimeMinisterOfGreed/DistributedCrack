#pragma once
#include "Async.hpp"
#include "Async/Executor.hpp"
#include "Concepts.hpp"
#include <functional>
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

struct ParallelLoopBuilder;

template <typename... Args> struct ParallelLoop : public Future<void, Args...> {
  friend struct ParallelLoopBuilder;

protected:
  std::tuple<Args...> _args;
  std::vector<sptr<Task>> _tasks;
  std::vector<std::function<void(Args...)>> _fncs;
  void set_args(Args... args) { _args = std::tuple<Args...>{args...}; }
  int iterations = 1;
  template <typename... Fs>
  ParallelLoop(Fs... f)
      : _fncs(std::vector<std::function<void(Args...)>>{f...}) {}

public:
  virtual void operator()() {
    for (int i = 0; i < iterations; i++) {
      for (auto f : _fncs) {
        std::apply(
            [this, f](Args... args) {
              auto t = Future<void, Args...>::Create(f, args...);
              _tasks.push_back(t);
            },
            _args);
      }
    }
  }
};

struct ParallelLoopBuilder {};