
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

template <typename T = void, typename... Args> class BasePromise : public Task {

protected:
  std::function<T(Args...)> _fnc;
  AsyncType type = START;
  std::tuple<Args...> _args;
  BasePromise(std::function<T(Args...)> fnc, AsyncType type, Args... args)
      : type(type), _fnc(fnc), _args(args...) {}

public:
  BasePromise(std::function<T(Args...)> fnc, Args... args)
      : BasePromise<T, Args...>{fnc, START, args...} {}

  void operator()() {
    using ret = T;
    if constexpr (std::is_void_v<ret>) {
      std::apply(_fnc, _args);
    } else {
      this->_result.emplace(std::apply(_fnc, _args));
    }
  }
};

template <typename T = void, typename... Args>
class Promise : public BasePromise<T, Args...> {

  using F = std::function<T(Args...)>;

protected:
  AsyncType _type = START;
  Task *_father = nullptr;

  Promise(Task *father, F fnc, Args... args)
      : Promise<T, Args...>(fnc, args...) {
    _father = father;
  }

public:
  Promise(F fnc, Args... args) : BasePromise<T, Args...>(fnc, args...) {
    Scheduler::main().schedule(this);
  }
};