#include "Async/Executor.hpp"
#include "MultiThread/AutoResetEvent.hpp"
#include <boost/smart_ptr/intrusive_ptr.hpp>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

Scheduler *Scheduler::_instance = new Scheduler();

int Scheduler::WaitAnyExecutorIdle() {
  std::vector<WaitHandle *> _handlers{};
  for (auto e : _executors)
    _handlers.push_back(&e->onCompleted);
  auto res = WaitAny(_handlers);
  return res;
}

bool Scheduler::assign(boost::intrusive_ptr<Task> task) {
  for (auto e : _executors) {
    if (e->assign(task))
      return true;
  }
  return false;
}

boost::intrusive_ptr<Task> Scheduler::post(std::function<void()> f) {
  std::lock_guard lock{schedLock};
  auto alloc = boost::intrusive_ptr<Task>{new PostableTask{f}};
  mq.push(alloc);
  return alloc;
}

void Scheduler::schedule(boost::intrusive_ptr<Task> task) {
  std::lock_guard lock{schedLock};
  mq.push(task);
}

std::optional<boost::intrusive_ptr<Task>> Scheduler::take() {
  std::lock_guard lock{schedLock};
  if (mq.size() == 0)
    return {};
  auto v = mq.front();
  mq.pop();
  return {v};
}

bool Scheduler::empty() const {
  if (mq.empty()) {
    for (auto &exec : _executors) {
      if (exec->state() != Executor::IDLE) {
        return false;
      }
    }
  }
  return false;
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
    auto ecount = _executors.size();
    if (ecount == 0) {
      _executors.push_back(boost::intrusive_ptr<Executor>(new Executor()));
      _executors.at(0)->start();
    }
    auto task = take();
    if (!task.has_value())
      continue;
    if (task.value()->state() == Task::RESOLVED)
      continue;
    else if (assign(task.value())) {
      continue;
    } else if (ecount < _maxExecutors) {
      auto executor = boost::intrusive_ptr<Executor>(new Executor());
      executor->start();
      _executors.push_back(executor);
      executor->assign(task.value());
    } else {
      auto free = WaitAnyExecutorIdle();
      _executors.at(free)->assign(task.value());
    }
  }
}

void Scheduler::stop() {}

void Scheduler::reset() {
  std::lock_guard lock{schedLock};
  while (!mq.empty()) {
    mq.pop();
  }
  for (auto ex : _executors) {
    ex->stop();
    ex->wait_termination();
  }
}

Scheduler &Scheduler::main() { return *_instance; }

Scheduler::Scheduler() {}

Scheduler::~Scheduler() {
  _end = true;
  reset();
}

Executor::Executor() {}

bool Executor::assign(boost::intrusive_ptr<Task> task) {
  if (status == PROCESSING || status == WAITING_EXECUTION)
    return false;
  _currentExecution.emplace(task);
  status = WAITING_EXECUTION;
  onAssigned.Set();
  return true;
}

void Executor::start() {
  if (_executingThread != nullptr)
    return;
  _executingThread = new std::thread{[this]() {
    while (!_end) {
      if (!_currentExecution.has_value()) {
        status = IDLE;
        onAssigned.WaitOne();
      }
      onCompleted.Reset();
      onAssigned.Reset();
      auto t = _currentExecution.value();
      status = PROCESSING;
      (*t)();
    }
    status = IDLE;
    onCompleted.Set();
  }};
  _executingThread->detach();
}

void Executor::wait_termination() { onCompleted.WaitOne(); }

void Executor::stop() { _end = true; }

Executor::~Executor() { _end = true; }

void intrusive_ptr_add_ref(Task *p) {}

void intrusive_ptr_add_ref(Executor *p) {}

void intrusive_ptr_release(Executor *p) {}
void intrusive_ptr_release(Task *p) {}

void Task::cancel() { resolve(); }

void Task::resolve(bool failed) {
  _state = failed ? FAILED : RESOLVED;
  _executed.Set();
  if (failed && _failureHandler != nullptr)
    Scheduler::main().schedule(_failureHandler);
  else if (_thenHandler != nullptr) {
    _thenHandler->_father = this;
    Scheduler::main().schedule(_thenHandler);
  }
}
void Task::set_then(boost::intrusive_ptr<Task> task) {
  _thenHandler = task;
  if (_state == RESOLVED) {
    _thenHandler->_father = this;
    Scheduler::main().schedule(_thenHandler);
  }
}
void Task::set_failure(boost::intrusive_ptr<Task> task) {
  _failureHandler = task;
  if (_state == FAILED) {
    Scheduler::main().schedule(_failureHandler);
  }
}
