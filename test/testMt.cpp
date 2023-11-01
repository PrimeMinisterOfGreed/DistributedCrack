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
