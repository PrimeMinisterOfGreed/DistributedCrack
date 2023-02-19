#include "LogEngine.hpp"
#include "MultiThread/MultiThreadStringGenerator.hpp"
#include "MultiThread/ThreadSchema.hpp"
#include "md5.hpp"
#include <gtest/gtest.h>

TEST(test_mt_schema, TestMtSchema)
{
    ConsoleLogEngine logEngine{3};
    ThreadMultiSchema schema{16, 4, md5("%%!%"), 2000, &logEngine};
    schema.Initialize();
    schema.ExecuteSchema();
    schema.GetResult();
}