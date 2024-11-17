#include "TestEnvironment.hpp"
#include <cstdio>
#include <gtest/gtest.h>
#include "MultiThread/functions.hpp"
#include "md5.hpp"

TEST(TestMt, test_chunk_function){
    auto chunk = std::vector<std::string>{"hello","world","!"};
    auto target = MD5(chunk[1]).hexdigest();
    auto result = compute_chunk(chunk, target, 3);
    ASSERT_EQ(result, 1);
}