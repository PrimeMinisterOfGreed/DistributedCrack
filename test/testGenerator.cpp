#include <gtest/gtest.h>
#include "StringGenerator.hpp"
#include "Functions.hpp"

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
	AssignedSequenceGenerator generator{ 1 };
	generator.AssignAddress(94);
	auto seq = generator.nextSequence();
	assert(seq.at(seq.size() - 1) == (char)minCharInt);
	assert(seq.at(seq.size() - 2) == (char)minCharInt + 1);
}

TEST(TestGenerator, test_indexof)
{
	std::vector<int> integers{ {0,1,2,3,4} };
	int index = indexOf<int>(integers.begin(), integers.end(), [=](int x) { return x == 4; });
	assert(index == 4);

}