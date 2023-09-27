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

TEST_F(MtTest, test_mt_calculation) {
  AutoResetEvent found{false};
  Promise<void>{[]() {
    bool end = false;
    AssignedSequenceGenerator generator{4};
    while (!end) {
      ManualResetEvent finished{false};
      std::string result = "";
      Promise<std::vector<std::string>>{[&generator]() {
        return generator.generateChunk(2000);
      }}.then<void>([&result, &finished](std::vector<std::string> vec) {
        for (int i = 0; i < vec.size(); i += 10) {
          Promise<void>{[&vec, &result, i, &finished] {
            for (int k = i; k <= i && k < vec.size(); k++) {
              if (md5("0000") == md5(vec[k])) {
                finished.Set();
                result = vec[k];
              }
              if (k == vec.size()) {
                finished.Set();
              }
            }
          }};
        }
      });
      finished.WaitOne();
      if (result != "") {
        end = true;
      }
    }
  }}.wait();
}