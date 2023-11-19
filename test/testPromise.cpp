#include "Async/Async.hpp"
#include "Async/AsyncLoop.hpp"
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
#include <cstdlib>
#include <cstring>
#include <functional>
#include <gtest/gtest.h>
#include <md5.hpp>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>

class TestPromise : public TestEnvironment {};

Scheduler &sched() { return Scheduler::main(); }

TEST_F(TestPromise, test_future) {
  sched().start();
  auto p = Future<int>::Run([]() { return 1; });
  int a = *p;
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

TEST_F(TestPromise, test_future_void) {
  int a = 0;
  int b = 0;
  auto p = async([&a]() { a = 4; });
  *p += [&b]() { b = 1; };
  sched().start();
  p->wait();
  ASSERT_EQ(4, a);
  ASSERT_EQ(1, b);
  sched().stop();
}

TEST_F(TestPromise, test_future_fnc) {
  int b = 0;
  auto p = async([]() { return 1; });
  sched().start();
  int a = *p;
  ASSERT_EQ(a, 1);
  sched().stop();
}

TEST(TestAlternateHandler, test_f) {
  Func f{[](int a, int b) { return a + b; }, 1, 2};
  f();
  int c = f.result().reintepret<int>();
  ASSERT_EQ(3, c);
}

TEST(TestAlternateHandler, test_f_ref) {
  int a = 0;
  Func f{[&a]() { a = 1; }};
  f();
  ASSERT_EQ(1, a);
}

TEST(TestAlternateHandler, test_f_lazy) { int a = 0; }