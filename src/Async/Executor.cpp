#include "Async/Executor.hpp"
#include "Macro.hpp"
#include <exception>
#include <thread>

Scheduler *Scheduler::_instance = new Scheduler();

void Scheduler::schedule(Task *task) { mq.push(task); }

void Scheduler::start() {}

void Scheduler::stop() {}

Scheduler &Scheduler::main() { return *_instance; }

Executor::Executor() {}

void Executor::assign(Task *task) {
  if (status == IDLE) {
    mq.push(task);
    auto t = std::thread{[this]() {
      while (!_end && mq.size() > 0) {
        Task *t = mq.front();
        status = PROCESSING;
        (*t)();
        mq.pop();
        status = WAITING_EXECUTION;
      }
      status = IDLE;
    }};
    t.detach();
    _executingThread.swap(t);
  } else {
    mq.push(task);
  }
}

Executor::~Executor() { _end = true; }
