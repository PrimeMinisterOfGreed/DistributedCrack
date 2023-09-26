#include "Async/Executor.hpp"
#include "LogEngine.hpp"
#include "MultiThread/MultiThreadStringGenerator.hpp"
#include "Statistics/TimeMachine.hpp"
#include "TestEnvironment.hpp"
#include "md5.hpp"
#include <chrono>
#include <cstdio>
#include <gtest/gtest.h>
#include <string>

struct MtTest : public TestEnvironment {};

TEST_F(MtTest, test_mt_generation) {}