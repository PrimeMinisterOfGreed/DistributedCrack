#include "Functions.hpp"
#include "StringGenerator.hpp"
#include "md5.hpp"
#include <cassert>
#include <cstdio>
#include <gtest/gtest.h>
#include <string>

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
    printf("original %s", "4a7d1ed414474e4033ac29ccb8653d9b");
    assert(md5("0000") == "4a7d1ed414474e4033ac29ccb8653d9b");
}

TEST(testMd5, test_gpu_md5)
{
}