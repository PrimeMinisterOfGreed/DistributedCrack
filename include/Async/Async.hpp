#pragma once
#include "Executor.hpp"
#include <boost/smart_ptr/intrusive_ptr.hpp>
#include <cstddef>
#include <functional>
#include <mutex>
#include <tuple>
#include <vector>

template <typename T, typename... Args> struct BaseAsync : public Task {
  using F = std::function<T(Args...)>;
  F _fnc;
  std::tuple<Args...> _args;
  BaseAsync(std::function<T(Args...)> &&fnc, Args... args)
      : _fnc(std::move(fnc)), _args(args...) {}
  BaseAsync(std::function<T(Args...)> &&fnc, boost::intrusive_ptr<Task> father)
      : _fnc(std::move(fnc)) {
    _father = father;
  }
  BaseAsync(BaseAsync &) = delete;

  void set_args(Args... args) { _args = std::tuple<Args...>{args...}; }

  virtual void operator()() override {
    if (_father != nullptr) {
      if (!(_father->state() == RESOLVED)) {
        Scheduler::main().schedule(this);
        return;
      }
      _result.emplace(_fnc(_father->result()));
      resolve();
    } else {
      _result.emplace(std::apply(_fnc, _args));
    }
  }
};

template <typename T> struct BaseAsync<T, void> : public Task {
  using F = std::function<T()>;
  F _fnc;
  BaseAsync(std::function<T()> &&fnc) : _fnc(std::move(fnc)) {}
  BaseAsync(BaseAsync &) = delete;
  virtual void operator()() override { _result.emplace(_fnc); }
};

template <typename... Args> struct BaseAsync<void, Args...> : public Task {
  using F = std::function<void(Args...)>;
  F _fnc;
  std::tuple<Args...> _args;
  BaseAsync(BaseAsync &) = delete;
  BaseAsync(std::function<void(Args...)> &&fnc, Args... args)
      : _fnc(std::move(fnc)), _args(args...) {}
  BaseAsync(std::function<void(Args...)> &&fnc,
            boost::intrusive_ptr<Task> father)
      : _fnc(std::move(fnc)) {
    _father = father;
  }

  void set_args(Args... args) { _args = std::tuple<Args...>{args...}; }

  virtual void operator()() {
    if (_father != nullptr) {
      if (!(_father->state() == RESOLVED)) {
        Scheduler::main().schedule(this);
        return;
      }
      if constexpr (sizeof...(Args) > 0) {
        _fnc(_father->result().reintepret<Args...>());
      } else {
        _fnc();
      }
    } else {
      std::apply(_fnc, _args);
    }
  }
};

template <typename T, typename... Args>
BaseAsync(std::function<T(Args...)> &&fnc, Args... args)
    -> BaseAsync<T, Args...>;

template <typename T, typename... Args>
BaseAsync(std::function<T()> &&fnc) -> BaseAsync<T>;

template <typename T> BaseAsync(std::function<T()> &&fnc) -> BaseAsync<T>;

template <typename T = void, typename... Args> struct Async {
  boost::intrusive_ptr<Task> actual{};

  Async(std::function<T(Args...)> &&fnc, Args... args) {
    auto alloc = boost::intrusive_ptr<Task>(
        new BaseAsync<T, Args...>(std::move(fnc), args...));
    actual = alloc;
    Scheduler::main().schedule(alloc);
  }

  Async(std::function<T(Args...)> &&fnc, boost::intrusive_ptr<Task> father) {
    auto alloc = boost::intrusive_ptr<Task>(
        new BaseAsync<T, Args...>(std::move(fnc), father));
    actual = alloc;
    alloc->set_father(father);
    father->set_children(alloc);
    Scheduler::main().schedule(alloc);
  }

  template <typename K> Async then(auto &&fnc) {
    return Async{std::move(fnc), actual};
  }

  Async fail(auto &&fnc) {
    auto alloc =
        boost::intrusive_ptr<Task>(new BaseAsync<void>{std::move(fnc)});
    actual->set_failure(alloc);
  }

  void wait() { actual->wait(); }
};

template <typename T, typename... Args>
Async(std::function<T(Args...)> &&) -> Async<T, Args...>;

template <typename T, typename... Args> struct AsyncLoop : public Task {
  std::function<T(Args...)> _iterFnc;
  std::function<bool(T)> _terminator;
  std::tuple<Args...> _args;
  AsyncLoop(std::function<bool(T)> predicate, std::function<T(Args...)> iterFnc,
            Args... args)
      : _terminator(predicate), _iterFnc(iterFnc), _args(args...) {
    Scheduler::main().schedule(boost::intrusive_ptr<Task>{this});
  }

  virtual void operator()() {
    T res = std::apply(_iterFnc, _args);
    if (_terminator(res)) {
      resolve();
    } else {
      Scheduler::main().schedule(this);
    }
  }
};

template <> struct AsyncLoop<void> : public Task {
  std::function<void(boost::intrusive_ptr<Task>)> _iterFnc;
  AsyncLoop(std::function<void(boost::intrusive_ptr<Task>)> iterFnc)
      : _iterFnc(iterFnc) {
    Scheduler::main().schedule(boost::intrusive_ptr<Task>(this));
  }
  virtual void operator()() {
    auto itrPtr = boost::intrusive_ptr<Task>(this);
    _iterFnc(itrPtr);
    Scheduler::main().schedule(itrPtr);
  }
};

struct AsyncMultiLoop : public Task {
  std::function<void(size_t)> _iterLambda;
  std::mutex _lock{};
  size_t _iterations;
  AsyncMultiLoop(size_t iterations, std::function<void(size_t)> iterLambda);
  virtual void operator()();
};