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

template <typename> class BaseFuture;

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

  virtual void operator()(sptr<Task> thisptr) override {
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
  static sptr<Future<T, Args...>> Create(F &&fnc, Args... args) {
    return sptr<Future<T, Args...>>{new Future<T, Args...>(fnc, args...)};
  }

  template <typename F>
  static sptr<Future<T, Args...>> Run(F &&fnc, Args... args) {
    auto ptr = Create(fnc, args...);
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

  template <typename F>
  static sptr<Future<void, Args...>> Create(F &&fnc, Args... args) {
    return sptr<Future<void, Args...>>{new Future<void, Args...>(fnc, args...)};
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
