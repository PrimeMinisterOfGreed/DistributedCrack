#include "Async/Async.hpp"
#include "StringGenerator.hpp"
#include "TestEnvironment.hpp"
#include "md5.hpp"
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <gtest/gtest.h>
#include <optional>
#include <string>
#include <vector>

struct MtTest : public TestEnvironment {};

void compute(AutoResetEvent &evt) {}

TEST_F(MtTest, test_mt_calculation) {
  AssignedSequenceGenerator generator{4};
  std::string result = "";
  while (result == "") {
    auto vec = generator.generateChunk(1000);
    AsyncMultiLoop{vec.size(),
                   [&vec, &result](size_t it) {
                     if (md5(vec[it]) == md5("!!!!")) {
                       result = vec[it];
                     }
                   }}
        .wait();
  }
  ASSERT_STREQ("0000", result.c_str());
}