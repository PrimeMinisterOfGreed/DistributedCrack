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

template <typename... Args> struct Async;

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
  Async(Args... args) {
    if constexpr (sizeof...(args) > 0) {
      _actual->make_data<std::tuple<Args...>>(std::tuple(args...));
    }
  }
  Async(sptr<Task> actual) : BaseAsync(actual) {}

  template <typename F>
    requires(is_void_function<F, Args...> || is_valid_async<F, Args...>)
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
    then_impl(task);
    _actual = task;
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

template <typename> struct BaseFuture;

template <typename T, typename... Args>
class BaseFuture<T(Args...)> : public Task {

  std::function<T(Args...)> _fnc;
  std::tuple<Args...> _args;

public:
  template <typename F> BaseFuture(F &&fnc, Args... args) : Task(), _fnc(fnc) {
    using ret_t = decltype(std::forward<F>(fnc)(std::declval<Args &>()...));
    static_assert(std::is_same_v<ret_t, T>,
                  "cannot use function with different return type");
    _fnc = fnc;
    if constexpr (sizeof...(Args) > 0)
      _args = {args...};
  }

  virtual void operator()() override {
    if constexpr (sizeof...(Args) > 0) {
      if constexpr (std::is_void_v<T>) {
        std::apply(_fnc, _args);
        resolve();
      } else {
        _result.emplace(std::apply(_fnc, _args));
        resolve();
      }
    } else {
      if constexpr (std::is_void_v<T>) {
        _fnc();
        resolve();
      } else {
        _result.emplace(_fnc());
        resolve();
      }
    }
  }

  T result() {
    wait();
    return _result.reintepret<T>();
  }

  operator T() { return result(); }
};

template <typename T, typename... Args>
class Future : public BaseFuture<T(Args...)> {
private:
  std::vector<std::function<void(T)>> _handlers;

public:
  template <typename F>
  Future(F &&fnc, Args... args) : BaseFuture<T(Args...)>(fnc, args...) {}

  template <typename F>
  static sptr<Future<T, Args...>> Run(F &&fnc, Args... args) {
    auto ptr = sptr<Future<T, Args...>>{new Future<T, Args...>(fnc, args...)};
    Scheduler::main().schedule(std::static_pointer_cast<Task>(ptr));
    return ptr;
  }
  virtual void resolve(bool failed = false) override {
    if (_handlers.size() > 0) {
      for (auto h : _handlers) {
        h(Task::_result.reintepret<T>());
      }
    }
    Task::resolve(failed);
  }
  template <typename F> void operator+=(F &&onCompletedHandler) {
    _handlers.push_back(onCompletedHandler);
  }
};

template <typename... Args>
class Future<void, Args...> : public BaseFuture<void(Args...)> {
private:
  std::vector<std::function<void()>> _handlers;

public:
  template <typename F>
  Future(F &&fnc, Args... args) : BaseFuture<void(Args...)>(fnc, args...) {}

  template <typename F>
  static sptr<Future<void, Args...>> Run(F &&fnc, Args... args) {
    auto ptr =
        sptr<Future<void, Args...>>{new Future<void, Args...>(fnc, args...)};
    Scheduler::main().schedule(std::static_pointer_cast<Task>(ptr));
    return ptr;
  }

  virtual void resolve(bool failed = false) override {
    if (_handlers.size() > 0) {
      for (auto h : _handlers) {
        h();
      }
    }
    Task::resolve(failed);
  }
  template <typename F> void operator+=(F &&onCompletedHandler) {
    _handlers.push_back(onCompletedHandler);
  }
};

template <typename F, typename... Args> auto async(F &&fnc, Args... args) {
  using ret_t = decltype(std::forward<F>(fnc)(std::declval<Args &>()...));
  auto ptr = Future<ret_t, Args...>::Run(fnc, args...);
  return ptr;
};
