#include "Async/Async.hpp"
#include "Async/Executor.hpp"
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

Scheduler &sched() { return Scheduler::main(); }

TEST(TestAsync, test_async_execution) {
  int a = 0;
  BaseAsyncTask b = BaseAsyncTask<void()>{[&a]() { a++; }};
  b();
  ASSERT_EQ(a, 1);
}

TEST(TestAsync, test_async_args) {
  int a = 0;
  BaseAsyncTask b = BaseAsyncTask<void(int *)>{[](int *c) { (*c)++; }, &a};
  b();
  ASSERT_EQ(a, 1);
}

TEST_F(TestPromise, test_async_start) {
  Scheduler::main().start();
  int a = 0;
  Async<>().start([&a]() { a++; }).wait();
  Scheduler::main().stop();
}

TEST_F(TestPromise, test_async_then) {
  int a = 0;
  Scheduler::main().start();
  Async<>()
      .start([&a]() {
        a++;
        return Async<int>(a);
      })
      .then([&a](int val) { a += 2; })
      .wait();
  Scheduler::main().stop();
}

TEST_F(TestPromise, test_future) {
  sched().start();
  auto p = Future<int>::Run([]() { return 1; });
  int a = p->result();
  sched().stop();
  ASSERT_EQ(a, 1);
}

int calc(int p) { return p + 1; }

TEST_F(TestPromise, test_future_ptr) {
  sched().start();
  auto p = Future<int, int>::Run(calc, 3);
  int a = *p;
  sched().stop();
  ASSERT_EQ(a, 4);
}

TEST_F(TestPromise, test_future_fnc) {
  int b = 0;
  auto p = async([]() { return 1; });
  *p += [&b](int t) { b += t; };
  sched().start();
  int a = *p;
  ASSERT_EQ(a, 1);
  ASSERT_EQ(1, b);
  sched().stop();
}

/*
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
*/