
#include "Async/Executor.hpp"
#include <gtest/gtest.h>

class TestEnvironment : public testing::Test {
public:
  virtual void SetUp() { Scheduler::main().start(); }

  virtual void TearDown() { Scheduler::main().reset(); }
};