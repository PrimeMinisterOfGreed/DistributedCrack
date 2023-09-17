
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

template <typename T = void, typename... Args> class Executable : public Task {

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

public:
  Promise(F fnc) {
    auto alloc = new Executable<T, Args...>{fnc};
    lastTask = alloc;
    Scheduler::main().schedule(alloc);
  }

  template <typename K = void> Promise &&then(std::function<K(T)> fnc) {}
};