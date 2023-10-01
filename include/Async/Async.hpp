#pragma once
#include "Executor.hpp"
#include <boost/smart_ptr/intrusive_ptr.hpp>
#include <functional>
#include <tuple>

template <typename T, typename... Args> struct BaseAsync : public Task {
  using F = std::function<T(Args...)>;
  F _fnc;
  std::tuple<Args...> _args;
  BaseAsync(std::function<T(Args...)> &&fnc, Args... args)
      : _fnc(std::move(fnc)), _args(args...) {}
  BaseAsync(std::function<T(Args...)> &&fnc, Task *father)
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
  BaseAsync(std::function<void(Args...)> &&fnc, Task *father)
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

template <typename T, typename... Args> struct Async {
  boost::intrusive_ptr<Task> actual{};

  Async(std::function<T(Args...)> &&fnc, Args... args) {
    auto alloc =
        boost::intrusive_ptr<Task>(new BaseAsync<T, Args...>(fnc, args...));
    actual = alloc;
    Scheduler::main().schedule(alloc);
  }

  Async(std::function<T(Args...)> &&fnc, Task *father) {
    auto alloc =
        boost::intrusive_ptr<Task>(new BaseAsync<T, Args...>(fnc, father));

    alloc->set_father(actual);
    actual->set_children(alloc);
    Scheduler::main().schedule(alloc);
  }

  template <typename K> Async &&then(std::function<K(T)> &&fnc) {
    return Async{fnc, actual};
  }

  Async &&fail(std::function<void()> &&fnc) {
    auto alloc = boost::intrusive_ptr<Task>();
    actual->set_failure(alloc);
  }
};
