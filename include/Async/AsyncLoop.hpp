#pragma once
#include "Async.hpp"
#include "Async/Executor.hpp"
#include "Concepts.hpp"
#include <functional>
#include <memory>
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

template <typename... Args> struct ParallelLoop : public Future<void, Args...> {

protected:
  std::tuple<Args...> _args;
  std::vector<sptr<Task>> _tasks;
  std::vector<std::function<void(Args...)>> _fncs;
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
              _tasks.push_back(t);
            },
            _args);
      }
    }
  }

  template <typename... Fs> static sptr<Task> Create(int it, Fs... f) {
    return std::make_shared(ParallelLoop<Args...>(it, f...));
  }
  template <typename... Fs> static sptr<Task> Create(Fs... f) {
    return Create(0, f...);
  }
};
