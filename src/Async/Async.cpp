#include "Async/Async.hpp"
#include "Async/Executor.hpp"
#include "MultiThread/AutoResetEvent.hpp"
#include <boost/smart_ptr/intrusive_ptr.hpp>
#include <cstddef>
#include <mutex>

AsyncMultiLoop::AsyncMultiLoop(size_t iterations,
                               std::function<void(size_t)> iterLambda)
    : _iterLambda(iterLambda), _iterations(iterations) {
  Scheduler::main().schedule(boost::intrusive_ptr<Task>{this});
}

void AsyncMultiLoop::operator()() {}
