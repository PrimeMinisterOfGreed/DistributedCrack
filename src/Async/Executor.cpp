#include "Async/Executor.hpp"
#include "Macro.hpp"
#include <cstddef>
#include <exception>
#include <mutex>
#include <optional>
#include <thread>

Scheduler *Scheduler::_instance = new Scheduler();

void Scheduler::schedule(Task *task) {
  std::lock_guard lock{schedLock};
  mq.push(task);
}

std::optional<Task *> Scheduler::take() {
  std::lock_guard lock{schedLock};
  if (mq.size() == 0)
    return {};
  auto *v = mq.front();
  mq.pop();
  return {v};
}

void Scheduler::start() {
  if (_executionThread == nullptr) {
    _executionThread = new std::thread{[this]() { routine(); }};
    _executionThread->detach();
  }
}

void Scheduler::routine() {
  while (!_end) {
    // TODO introduce an optimization to decide if disband or allocate new
    // executors
    if (_executors.size() == 0) {
      _executors.push_back(new Executor());
      _executors.at(0)->start();
    }
    auto task = take();
    if (!task.has_value())
      continue;
    if (AssignToIdle(task.value())) {
      continue;
    }
    if (AssignToLowerCharged(task.value())) {
      continue;
    }

    // measure and spawn a new thread
  }
}

void Scheduler::stop() {}

void Scheduler::reset() {
  std::lock_guard lock{schedLock};
  while (!mq.empty()) {
    auto *t = mq.front();
    delete t;
    mq.pop();
  }
  for (auto ex : _executors) {
    delete ex;
  }
}

Scheduler &Scheduler::main() { return *_instance; }

Scheduler::Scheduler() {}

Scheduler::~Scheduler() {
  _end = true;
  reset();
}

bool Scheduler::AssignToIdle(Task *task) {
  for (auto ex : this->_executors) {
    if (ex->state() == Executor::IDLE) {
      ex->assign(task);
      return true;
    }
  }
  return false;
}

bool Scheduler::AssignToLowerCharged(Task *task) { return false; }

std::optional<Task *> Executor::take() {
  std::lock_guard<std::mutex> lock{queueLock};
  if (mq.size() > 0) {
    auto v = mq.front();
    mq.pop();
    return {v};
  }
  return {};
}

void Executor::push(Task *task) {
  std::lock_guard<std::mutex> lock{queueLock};
  mq.push(task);
}

Executor::Executor() {}

void Executor::assign(Task *task) { push(task); }

void Executor::start() {
  if (_executingThread != nullptr)
    return;
  _executingThread = new std::thread{[this]() {
    while (!_end) {
      auto v = take();
      if (!v.has_value())
        continue;
      auto t = v.value();
      status = PROCESSING;
      (*t)();
      if (t->_children == nullptr) {
        delete t;
      }

      status = WAITING_EXECUTION;
    }
    status = IDLE;
  }};
  _executingThread->detach();
}

Executor::~Executor() { _end = true; }
