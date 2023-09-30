#include "TestEnvironment.hpp"
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