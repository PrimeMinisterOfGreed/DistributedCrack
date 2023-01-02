#include "Functions.hpp"
#include "StringGenerator.hpp"
#include "md5.hpp"
#include "md5_gpu.hpp"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <gtest/gtest.h>
#include <string>
#include <vector>

TEST(TestGenerator, test_known_sequence)
{
    AssignedSequenceGenerator generator{4};
    generator.AssignAddress(372);
    auto seq = generator.nextSequence();
    assert(seq.at(seq.size() - 1) == (char)minCharInt);
    assert(seq.at(seq.size() - 2) == (char)minCharInt + 4);
}

TEST(TestGenerator, test_sequence_increment)
{
    AssignedSequenceGenerator generator{1};
    generator.AssignAddress(93);
    auto seq = generator.nextSequence();
    assert(seq.at(seq.size() - 1) == (char)minCharInt);
    assert(seq.at(seq.size() - 2) == (char)minCharInt + 1);
}

TEST(TestGenerator, test_indexof)
{
    std::vector<int> integers{{0, 1, 2, 3, 4}};
    int index = indexOf<int>(integers.begin(), integers.end(), [=](int x) { return x == 4; });
    assert(index == 4);
}

TEST(testMd5, test_md5)
{
    std::string computed = md5("0000");
    printf("computed: %s\n", computed.c_str());
    printf("original %s\n", "4a7d1ed414474e4033ac29ccb8653d9b");
    assert(md5("0000") == "4a7d1ed414474e4033ac29ccb8653d9b");
}

TEST(testMd5, test_gpu_md5)
{
    auto chunk = std::vector<std::string>({"0000"});
    auto md5GpuDigest = md5_gpu(chunk, 1);
    MD5 md5cpu = MD5(chunk.at(0));
    auto md5CPU = md5cpu.hexdigest();
    ASSERT_EQ(md5CPU, hexdigest(md5GpuDigest).at(0));
}

TEST(testMd5, test_multiple_digests)
{
    auto chunk = std::vector<std::string>({{"0000"},{"0001"},{"0002"}});
    std::vector<std::string> & md5Gpu = hexdigest(md5_gpu(chunk,3));
    for (int i = 0; i < chunk.size(); i++)
    {
     ASSERT_EQ(md5Gpu[i], md5(chunk.at(i)));   
    }
}