#pragma once
#include "Async/Executor.hpp"
#include "Concepts.hpp"
#include <boost/smart_ptr/intrusive_ptr.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

template <typename> struct BaseAsyncTask;
struct BaseAsync;

template <typename F, typename... Args>
concept is_valid_async = is_void_function<F, Args...> ||
                         is_function_return<BaseAsync &&, F, Args...>;

template <typename T, typename... Args>
struct BaseAsyncTask<T(Args...)> : public Task {

  std::function<T(Args...)> _fnc;
  std::tuple<Args...> _args;

  template <typename F>
  BaseAsyncTask(F &&fnc, Args... args) : _fnc(std::move(fnc)), _args(args...) {}

  template <typename F>
  BaseAsyncTask(F &&fnc, sptr<Task> father) : _fnc(std::move(fnc)) {
    _father = father;
  }

  BaseAsyncTask(BaseAsyncTask &) = delete;

  void set_args(Args... args) { _args = std::tuple<Args...>{args...}; }

  virtual void operator()() override {
    const auto argsize = sizeof...(Args);
    const bool isvoid = std::is_void_v<T>;
    if (_father != nullptr) {
      if constexpr (isvoid) {
        if constexpr (argsize > 0)
          _fnc(_father->result().reintepret<Args...>());
        else
          _fnc();
      } else {
        if constexpr (argsize > 0) {
          _result.emplace(_fnc(_father->result().reintepret<Args...>()));
        } else {
          _result.emplace(_fnc);
        }
      }
      resolve();
    } else {
      if constexpr (isvoid) {
        if constexpr (argsize > 0)
          std::apply(_fnc, _args);
        else
          _fnc();
        resolve();
      } else {
        if constexpr (sizeof...(Args) > 0) {
          _result.emplace(std::apply(_fnc, _args));
        } else {
          _result.emplace(_fnc());
        }
        resolve();
      }
    }
  }
};

struct BaseAsync {
protected:
  sptr<Task> _actual;

public:
  BaseAsync() {}
  BaseAsync(sptr<Task> actual) : _actual(actual) {}
  BaseAsync(BaseAsync &) = delete;
  void start_impl(sptr<Task> task) { Scheduler::main().schedule(task); }

  void then_impl(sptr<Task> task) { _actual->set_then(task); }

  void fail_impl(sptr<Task> task) { task->set_failure(task); }
};

template <typename... Args> struct Async : BaseAsync {

public:
  Async() : BaseAsync() {}
  Async(sptr<Task> actual) : BaseAsync(actual) {}

  template <typename F>
    requires(std::is_invocable_v<F, Args...> && is_valid_async<F, Args...>)
  auto &&start(F &&fnc, Args... args) {
    using ret_t = decltype(std::forward<F>(fnc)(std::declval<Args &>()...));
    auto task = sptr<Task>(
        new BaseAsyncTask<ret_t(Args...)>(std::forward<F>(fnc), args...));
    _actual = task;
    start_impl(task);
    if constexpr (std::is_void_v<ret_t>) {
      return static_cast<Async &&>(*this);
    } else {

      return reinterpret_cast<ret_t &&>(*this);
    }
  }

  template <typename F>
    requires(is_void_function<F, Args...> || is_valid_async<F, Args...>)
  auto &&then(F &&fnc) {
    using ret_t = decltype(std::forward<F>(fnc)(std::declval<Args &>()...));
    auto task = sptr<Task>(
        new BaseAsyncTask<ret_t(Args...)>(std::forward<F>(fnc), _actual));
    _actual = task;
    then_impl(task);
    if constexpr (std::is_void_v<ret_t>) {
      return static_cast<Async &&>(*this);
    } else {
      return reinterpret_cast<ret_t &&>(*this);
    }
  }

  void wait() { _actual->wait(); }
};

template <typename T, typename... Args> struct AsyncLoop : public Task {
  std::function<T(Args...)> _iterFnc;
  std::function<bool(T)> _terminator;
  std::tuple<Args...> _args;
  AsyncLoop(std::function<bool(T)> predicate, std::function<T(Args...)> iterFnc,
            Args... args)
      : _terminator(predicate), _iterFnc(iterFnc), _args(args...) {
    Scheduler::main().schedule(std::shared_ptr<Task>{this});
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
  std::function<void(std::shared_ptr<Task>)> _iterFnc;
  AsyncLoop(std::function<void(std::shared_ptr<Task>)> iterFnc)
      : _iterFnc(iterFnc) {
    Scheduler::main().schedule(std::shared_ptr<Task>(this));
  }
  virtual void operator()() {
    auto itrPtr = std::shared_ptr<Task>(this);
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