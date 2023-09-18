#include "Async/Executor.hpp"
#include "Async/Promise.hpp"
#include "Compute.hpp"
#include "Concepts.hpp"
#include "EventHandler.hpp"
#include "LogEngine.hpp"
#include "MultiThread/AutoResetEvent.hpp"
#include "OptionsBag.hpp"
#include <StringGenerator.hpp>
#include <boost/mpi.hpp>
#include <boost/program_options/variables_map.hpp>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <gtest/gtest.h>
#include <md5.hpp>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

TEST(TestHandler, test_event_handler) {
  EventHandler<> handler{};
  handler += new FunctionHandler([]() {});
}

TEST(TestPromise, test_promise_execution) {
  int a = 0;
  auto p = Executable<>([&a]() mutable { a = 1; });
  p();
  ASSERT_EQ(a, 1);
}

TEST(TestPromise, test_argumented_promise) {
  int a = 0;
  auto p = Executable<void, int &>([](int &a) { a = 10; }, a);
  p();
  ASSERT_EQ(a, 10);
}

TEST(TestPromise, test_return) {
  auto p = Executable<int>([] { return 10; });
  p();
  auto value = p.result().reintepret<int>();
  ASSERT_EQ(value, 10);
}

TEST(TestPromise, test_promise) {
  int a = 10;
  Promise<>{[&a]() { a++; }};
  Task &p = *Scheduler::main().mq.front();
  p();
  ASSERT_EQ(11, a);
}

TEST(TestExecutor, test_deferred_execution) {
  int a = 10;
  AutoResetEvent event{false};
  Executor ex{};
  auto p = new Executable<>{[&a, &ex, &event]() {
    for (int i = 0; i < 10; i++) {
      auto e = new Executable<>{[&a, i, &event] {
        a++;
        if (i == 9)
          event.Set();
      }};
      ex.assign(e);
    }
  }};
  ex.assign(p);
  event.WaitOne();
  ASSERT_EQ(a, 20);
}

TEST(TestPromise, test_then) {
  int a = 0;
  Promise<>{[&a] { a++; }}.then<void>([&a]() { a++; });
  Task &p = *Scheduler::main().mq.front();
  p();
  Scheduler::main().mq.pop();
  Task &t = *Scheduler::main().mq.front();
  t();
  ASSERT_EQ(2, a);
};
