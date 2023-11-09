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
  static sptr<ParallelLoop<Args...>> Create(int it, Args... args, Fs... f) {
    auto ptr = sptr<ParallelLoop<Args...>>(new ParallelLoop<Args...>(it, f...));
    ptr->set_args(args...);
    Scheduler::main().schedule(ptr);
    return ptr;
  }
  template <typename... Fs> static auto Create(Fs... f) {
    return Create(1, f...);
  }

  ~ParallelLoop() { wait(); }
};
