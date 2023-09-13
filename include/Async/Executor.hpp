#pragma once
#include <memory>
#include <queue>

class Executor;

class BaseAsync {
  friend class Executor;

protected:
  void operator()();
};

class Executor {
  std::queue<std::unique_ptr<BaseAsync>> mq{};
  Executor();

public:
  void start();
  void stop();
  static Executor &main();
};