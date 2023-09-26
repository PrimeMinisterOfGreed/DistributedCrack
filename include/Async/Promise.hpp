
#pragma once
#include "Async/Executor.hpp"
#include <bits/utility.h>
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

template <typename T = void, typename... Args> class Executable : public Task {
  friend class Promise<T, Args...>;

protected:
  std::function<T(Args...)> _fnc;
  AsyncType type = START;
  std::tuple<Args...> _args;
  Executable(std::function<T(Args...)> fnc, AsyncType type, Args... args)
      : type(type), _fnc(fnc), _args(args...) {}

public:
  Executable(std::function<T(Args...)> fnc, Args... args)
      : Executable<T, Args...>{fnc, START, args...} {}

  Executable(std::function<T(Args...)> fnc, Task *father) : _fnc(fnc) {
    _father = father;
  }

  void operator()() {
    using ret = T;
    if (_father != nullptr) {
      if (_father->state() != RESOLVED) {
        (*_father)();
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
  virtual ~Executable() {
    if (_father != nullptr) {
      delete _father;
    }
    Task::~Task();
  }
};

template <typename T = void, typename... Args> class Promise {
  using F = std::function<T(Args...)>;

private:
  Task *lastTask;

public:
  Promise(F fnc, Task *last) {
    auto alloc = new Executable<T, Args...>{fnc, last};
    Scheduler::main().schedule(alloc);
    last->set_children(alloc);
    lastTask = alloc;
  }

  Promise(F fnc) {
    auto alloc = new Executable<T, Args...>{fnc};
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
