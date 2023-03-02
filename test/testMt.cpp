#include "LogEngine.hpp"
#include "MultiThread/MultiThreadStringGenerator.hpp"
#include "MultiThread/ThreadSchema.hpp"
#include "Statistics/TimeMachine.hpp"
#include "md5.hpp"
#include <chrono>
#include <cstdio>
#include <gtest/gtest.h>
#include <string>

TEST(test_mt_schema, TestMtSchema)
{
    ConsoleLogEngine logEngine{3};
    ThreadMultiSchema schema{16, 4, md5("%%!%"), 2000, &logEngine};
    schema.Initialize();
    schema.ExecuteSchema();
    schema.GetResult();
}

TEST(test_possible_openmp, TestMpSchema)
{
    auto clock = std::chrono::steady_clock();
    ConsoleLogEngine testEngine(3);
    auto openmptime = executeTimeComparison([]() {
        MultiThreadStringGenerator generator(4);
        std::string password = "";
        while (password == "")
        {
            auto chunk = generator.generateChunk(100000);
#pragma omp parallel
            {
                for (int i = 0; i < chunk.size(); i++)
                {
                    if (md5(chunk.at(i)) == md5("!%%%"))
                    {
                        password = chunk.at(i);
                        break;
                    }
                }
            }
        }
    });

    auto schemaTime = executeTimeComparison([]() {
        ConsoleLogEngine logEngine{3};
        ThreadMultiSchema schema{16, 4, md5("!%%%"), 2000, &logEngine};
        schema.Initialize();
        schema.ExecuteSchema();
        schema.GetResult();
    });

    testEngine.TraceResult("Schema time {} than openmp time by {}", openmptime > schemaTime ? "faster" : "slower",
                           openmptime> schemaTime? (double)(openmptime/schemaTime)*100: (double)(schemaTime/openmptime)*100);
}