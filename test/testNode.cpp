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

class TestPromise : public testing::Test {
public:
  virtual void SetUp() { Scheduler::main().start(); }

  virtual void TearDown() {
    Scheduler::main().stop();
    Scheduler::main().reset();
  }
};

TEST(TestHandler, test_event_handler) {
  EventHandler<> handler{};
  handler += new FunctionHandler([]() {});
}

TEST(TestExecutable, test_promise_execution) {
  int a = 0;
  auto p = Executable<>([&a]() mutable { a = 1; });
  p();
  ASSERT_EQ(a, 1);
}

TEST(TestExecutable, test_argumented_promise) {
  int a = 0;
  auto p = Executable<void, int>([&a](int b) { a = b; }, 10);
  p();
  ASSERT_EQ(a, 10);
}

TEST(TestExecutable, test_return) {
  auto p = Executable<int>([] { return 10; });
  p();
  auto value = p.result().reintepret<int>();
  ASSERT_EQ(value, 10);
}

TEST(TestExecutable, test_promise) {
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

TEST(TestExecutable, test_then) {
  int a = 0;
  Promise<>{[&a] { a++; }}.then<void>([&a]() { a++; });
  Task &p = *Scheduler::main().mq.front();
  p();
  Scheduler::main().mq.pop();
  Task &t = *Scheduler::main().mq.front();
  t();
  ASSERT_EQ(2, a);
};

TEST(TestExecutable, test_promise_return) {
  int a = 0;
  Promise<int>{[]() { return 1; }}.then<int>([&a](int ns) {
    a = ns;
    return 0;
  });
  Task &p = *Scheduler::main().mq.front();
  p();
  Scheduler::main().mq.pop();
  Task &t = *Scheduler::main().mq.front();
  t();
  ASSERT_EQ(1, a);
}

TEST_F(TestPromise, test_auto_execution) {
  int a = 0;
  Promise<int>{[]() { return 10; }}.then([&a](int b) { a = b; }).wait();
  ASSERT_EQ(a, 10);
}

TEST_F(TestPromise, test_complex_execution) {
  int a = 0;
  Promise<int>{[]() { return 1 + 2; }}
      .then<int>([](auto a) { return a + 3; })
      .then<int>([](auto c) { return c + 3; })
      .then([&a](auto d) { a = d; })
      .wait();
  ASSERT_EQ(a, 9);
}