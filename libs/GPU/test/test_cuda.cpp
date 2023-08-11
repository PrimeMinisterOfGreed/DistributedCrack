#include <gtest/gtest.h>
#include <cuda.h>
#include <md5Cuda.cuh>
#include <support.cuh>

TEST(testGPU, test_ptr)
{
    char buffer[100]{};
    auto ptr = GpuPtr<char>(100);
    GpuMemSet(ptr(), 1, 100);
    ptr.copyTo(buffer);
    ASSERT_EQ(1, buffer[1]);
}

TEST(testGPU, test_ptr_compare)
{
    uint8_t buffer[100]{};
    for (uint8_t i = 0; i < 100; i++)
        buffer[i] = i;
    auto a = GpuPtr<uint8_t>{100};
    a.copyFrom(buffer);
    auto b = GpuPtr<uint8_t>{100};
    b.copyFrom(buffer);
    auto c = GpuPtr<uint8_t>{100};
    diff(a(),b(),c(),100);
    c.copyTo(buffer);
    ASSERT_EQ(2,buffer[1]);
}