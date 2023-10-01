
#pragma once
#include "Async/Executor.hpp"
#include <bits/utility.h>
#include <boost/smart_ptr/intrusive_ptr.hpp>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <tuple>
#include <type_traits>

template <typename T, typename... Args> class Promise;

template <typename T = void, typename... Args>
struct BasePromise : public Task {

  std::function<T(Args...)> _fnc;
  AsyncType type = START;
  std::tuple<Args...> _args;
  BasePromise(std::function<T(Args...)> fnc, AsyncType type, Args... args)
      : type(type), _fnc(fnc), _args(args...) {}

  BasePromise(std::function<T(Args...)> fnc, Args... args)
      : BasePromise<T, Args...>{fnc, START, args...} {}

  BasePromise(std::function<T(Args...)> fnc, boost::intrusive_ptr<Task> father)
      : _fnc(fnc) {
    _father = father;
  }

  void operator()() {
    using ret = T;
    if (_father != nullptr) {
      if (_father->state() != RESOLVED) {
        Scheduler::main().schedule(boost::intrusive_ptr<Task>{this});
        return;
      }
      if constexpr (sizeof...(Args) > 0) {
        if constexpr (std::is_void_v<T>) {
          _fnc(_father->result().reintepret<Args...>());
        } else {
          _result.emplace(_fnc(_father->result().reintepret<Args...>()));
        }
        _state = RESOLVED;
        resolve();
        return;
      }
    }
    if constexpr (std::is_void_v<ret>) {
      std::apply(_fnc, _args);
    } else {
      this->_result.emplace(std::apply(_fnc, _args));
    }
    _state = RESOLVED;
    resolve();
  }
};

template <typename T = void, typename... Args> class Promise {
  using F = std::function<T(Args...)>;

private:
  boost::intrusive_ptr<Task> lastTask{};

public:
  Promise(F fnc, boost::intrusive_ptr<Task> last) {
    auto alloc =
        boost::intrusive_ptr<Task>{new BasePromise<T, Args...>{fnc, last}};
    Scheduler::main().schedule(alloc);
    last->set_children(alloc);
    lastTask = alloc;
  }

  Promise(F fnc) {
    auto alloc = boost::intrusive_ptr<Task>{new BasePromise<T, Args...>{fnc}};
    lastTask = alloc;
    Scheduler::main().schedule(alloc);
  }

  template <typename K>
  Promise<K, T> then(auto fnc)
    requires(!std::is_void_v<T>)
  {
    return Promise<K, T>{fnc, lastTask};
  }

  template <typename K> Promise<K> then(std::function<K(void)> fnc) {
    return Promise<K>{fnc};
  }

  void wait() { lastTask->wait(); }
};
