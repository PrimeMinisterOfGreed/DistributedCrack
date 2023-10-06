#include "Async/Async.hpp"
#include "Async/Executor.hpp"
#include "Async/Promise.hpp"
#include "Concepts.hpp"
#include "LogEngine.hpp"
#include "MultiThread/AutoResetEvent.hpp"
#include "OptionsBag.hpp"
#include "TestEnvironment.hpp"
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

class TestPromise : public TestEnvironment {};

TEST(TestExecutable, test_promise_execution) {
  int a = 0;
  auto p = BasePromise<>([&a]() mutable { a = 1; });
  p();
  ASSERT_EQ(a, 1);
}

TEST(TestExecutable, test_argumented_promise) {
  int a = 0;
  auto p = BasePromise<void, int>([&a](int b) { a = b; }, 10);
  p();
  ASSERT_EQ(a, 10);
}

TEST(TestExecutable, test_return) {
  auto p = BasePromise<int>([] { return 10; });
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

TEST(TestExecutor, test_deferred_execution) {}

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
  Promise<int>{[]() { return 10; }}.then<void>([&a](int b) { a = b; }).wait();
  ASSERT_EQ(a, 10);
}

TEST_F(TestPromise, test_complex_execution) {
  int a = 0;
  Promise<int>{[]() { return 1 + 2; }}
      .then<int>([](auto a) { return a + 3; })
      .then<int>([](auto c) { return c + 3; })
      .then<void>([&a](auto d) { a = d; })
      .wait();
  ASSERT_EQ(a, 9);
}

TEST_F(TestPromise, test_nested_promise) {
  int a = 0;
  AutoResetEvent evt{false};
  Scheduler::main().post({[&a, &evt] {
    for (int i = 0; i < 100; i++) {
      if (i == 99) {
        Promise<>{[&a, &evt]() {
          a++;
          evt.Set();
        }};
      } else {
        Promise<>{[&a]() { a++; }};
      }
    }
  }});
  evt.WaitOne();
}

TEST(TestAsync, test_async_execution) {
  int a = 0;
  BaseAsync b = BaseAsync<void>{[&a]() { a++; }};
  b();
  ASSERT_EQ(a, 1);
}

TEST(TestAsync, test_async_args) {
  int a = 0;
  BaseAsync b = BaseAsync<void, int *>{[](int *c) { (*c)++; }, &a};
  b();
  ASSERT_EQ(a, 1);
}

TEST_F(TestPromise, test_async) {
  Async<void>{[]() {}}.then<void>([]() {}).wait();
}

TEST_F(TestPromise, test_async_loop) {
  int a = 0;
  AsyncLoop<int, int *>{[](int a) { return a == 100; },
                        [](int *a) {
                          (*a)++;
                          return *a;
                        },
                        &a}
      .wait();
  ASSERT_EQ(100, a);
}

TEST_F(TestPromise, test_async_void_loop) {
  int a = 0;
  AsyncLoop<void>{[&a](auto task) {
    a++;
    if (a == 100)
      task->cancel();
  }}.wait();
  ASSERT_EQ(100, a);
}

TEST_F(TestPromise, test_async_mt) {
  int a = 0;
  AsyncMultiLoop{100, [&a](size_t itr) { a = itr; }}.wait();
  ASSERT_TRUE(a != 0);
}