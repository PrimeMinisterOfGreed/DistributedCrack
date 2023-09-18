
#pragma once
#include "Async/Executor.hpp"
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <tuple>
#include <type_traits>

enum AsyncType { START, THEN, RESULT };

template <typename T, typename... Args> class Promise;

template <typename T = void, typename... Args> class Executable : public Task {
  friend class Promise<T, Args...>;

protected:
  std::function<T(Args...)> _fnc;
  AsyncType type = START;
  std::tuple<Args...> _args;
  Executable(std::function<T(Args...)> fnc, AsyncType type, Args... args)
      : type(type), _fnc(fnc), _args(args...) {}
  Task *_father;

public:
  Executable(std::function<T(Args...)> fnc, Args... args)
      : Executable<T, Args...>{fnc, START, args...} {}

  void operator()() {
    using ret = T;
    if constexpr (std::is_void_v<ret>) {
      std::apply(_fnc, _args);
    } else {
      this->_result.emplace(std::apply(_fnc, _args));
    }
  }
};

template <typename T = void, typename... Args> class Promise {
  using F = std::function<T(Args...)>;

private:
  Task *lastTask;

  Promise(F fnc, Task *last) {
    auto alloc = new Executable<T, Args...>{fnc};
    alloc->_father = last;
    Scheduler::main().schedule(alloc);
  }

public:
  Promise(F fnc) {
    auto alloc = new Executable<T, Args...>{fnc};
    lastTask = alloc;
    Scheduler::main().schedule(alloc);
  }

  template <typename K = void>
  Promise<K, T> &&then(auto fnc)
    requires(!std::is_void_v<T>)
  {
    return Promise<K, T>{fnc, lastTask};
  }

  template <typename K = void> Promise<K> &&then(std::function<K(void)> fnc) {
    return Promise<K>{fnc, lastTask};
  }
};