#include "Async/Executor.hpp"
#include "Async/Promise.hpp"
#include "LogEngine.hpp"
#include "MultiThread/AutoResetEvent.hpp"
#include "MultiThread/MultiThreadStringGenerator.hpp"
#include "Statistics/TimeMachine.hpp"
#include "StringGenerator.hpp"
#include "TestEnvironment.hpp"
#include "md5.hpp"
#include <chrono>
#include <cstdio>
#include <gtest/gtest.h>
#include <string>
#include <vector>

struct MtTest : public TestEnvironment {};

void compute(AutoResetEvent &evt) {}

TEST_F(MtTest, test_mt_calculation) {
  AutoResetEvent found{false};

  found.WaitOne();
}