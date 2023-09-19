#include "Async/Executor.hpp"
#include "Macro.hpp"
#include <cstddef>
#include <exception>
#include <thread>

Scheduler *Scheduler::_instance = new Scheduler();

void Scheduler::schedule(Task *task) { mq.push(task); }

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
    }
    auto task = mq.front();
    if (task == nullptr)
      continue;
    if (AssignToIdle(task)) {
      mq.pop();
      continue;
    }
    if (AssignToLowerCharged(task)) {
      mq.pop();
      continue;
    }

    // measure and spawn a new thread
  }
}

void Scheduler::stop() {}

void Scheduler::reset() {
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

Scheduler::Scheduler() { _executors.push_back(new Executor{}); }

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

bool Scheduler::AssignToLowerCharged(Task *task) {}

Executor::Executor() {}

void Executor::assign(Task *task) {
  if (status == IDLE) {
    if (_executingThread != nullptr) {
      delete _executingThread;
      _executingThread = nullptr;
    }
    mq.push(task);
    _executingThread = new std::thread{[this]() {
      while (!_end && mq.size() > 0) {
        Task *t = mq.front();
        status = PROCESSING;
        (*t)();
        if (t->_children == nullptr) {
          delete t;
        }
        mq.pop();
        status = WAITING_EXECUTION;
      }
      status = IDLE;
    }};
    _executingThread->detach();
  } else {
    mq.push(task);
  }
}

Executor::~Executor() { _end = true; }
