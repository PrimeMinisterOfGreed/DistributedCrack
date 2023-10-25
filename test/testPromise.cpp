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
  int a = 0;
  Async<>().start([&a]() { a++; }).wait();
}

TEST_F(TestPromise, test_async_then) {
  int a = 0;
  Async<>()
      .start([&a]() {
        a++;
        return a;
      })
      .then([&a](int val) { a += 2; })
      .wait();
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