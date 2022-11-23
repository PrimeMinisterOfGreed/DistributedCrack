#include <gtest/gtest.h>
#include "StringGenerator.hpp"

TEST(testGenerator, test_wrong_address)
{
	AssignedSequenceGenerator gen(4);
	
	try
	{
		gen.AssignAddress(2000);
		gen.AssignAddress(0);
		FAIL();
	}
	catch (const std::exception & e)
	{
		std::cout << e.what() << std::endl;
	}
}

