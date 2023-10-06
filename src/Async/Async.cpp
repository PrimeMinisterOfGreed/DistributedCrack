#include "Async/Async.hpp"
#include "Async/Executor.hpp"
#include "MultiThread/AutoResetEvent.hpp"
#include <boost/smart_ptr/intrusive_ptr.hpp>
#include <mutex>

AsyncMultiLoop::AsyncMultiLoop(int iterations,
                               std::function<void(size_t)> iterLambda)
    : _iterLambda(iterLambda), _iterations(iterations) {
  Scheduler::main().schedule(boost::intrusive_ptr<Task>{this});
}

void AsyncMultiLoop::operator()() {
  int limit = _iterations;
  for (int i = 0; i < limit; i++) {
    Async<void>{[i, this]() { _iterLambda(i); }}.then<void>([this]() {
      std::lock_guard l{_lock};
      _iterations--;
      if (_iterations == 0)
        resolve();
    });
  }
}
