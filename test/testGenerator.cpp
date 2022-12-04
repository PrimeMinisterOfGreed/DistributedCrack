#include "Functions.hpp"
#include "Statistics/CsvManager.hpp"
#include "StringGenerator.hpp"
#include <cassert>
#include <gtest/gtest.h>

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

TEST(TestCsvManager, test_save)
{
    CsvManager manager{"testFile.csv"};
    manager.Save({"prova", "ciao"}, RowIterator<double>{{{0, 1}, {1, 0}}});
}

TEST(TestCsvManager, test_load)
{
    CsvManager manager{"testFile.csv"};
    auto result = manager.Load();
    auto headers = result.first;
    auto rows = result.second;
    assert(headers.at(0) == "prova");
    assert(headers.at(1) == "ciao");
}