#include "LogEngine.hpp"
#include "MultiThread/ThreadSchema.hpp"
#include "Statistics/TimeMachine.hpp"
#include "StringGenerator.hpp"
#include "md5_gpu.hpp"
#include <cstddef>
#include <cstdio>
#include <gtest/gtest.h>
#include <string>

TEST(TestMtVsGpu, testmd5GpuvsmtNormalChunk)
{
    int chunkSize = 2000;
    ConsoleLogEngine testLogger(4);
    auto mtTime = executeTimeComparison([chunkSize]() {
        ConsoleLogEngine logEngine{3};
        ThreadMultiSchema schema{16, 4, md5("!%%%"), chunkSize, &logEngine};
        schema.Initialize();
        schema.ExecuteSchema();
        schema.GetResult();
    });

    auto gpuTime = executeTimeComparison([chunkSize]() {
        AssignedSequenceGenerator generator(4);
        size_t result = -1;
        while (result == -1)
        {
            auto &chunks = generator.generateChunk(chunkSize);
            result = md5_gpu(chunks, 256, "!%%%");
            if (result != -1)
                printf("password is %d\n", chunks.at(result).c_str());
            chunks.clear();
        }
    });

    testLogger.TraceResult("Gpu {} than Cpu by {}%", gpuTime>mtTime?"slower":"faster", gpuTime>mtTime? (double)(gpuTime/mtTime)*100: (double)(mtTime/gpuTime)*100);
    
}