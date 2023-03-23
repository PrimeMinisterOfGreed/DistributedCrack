#include <future>
#include <gtest/gtest.h>
#include "MultiThread/AutoResetEvent.hpp"

TEST(TestResettableEvents, test_wait_any)
{
    AutoResetEvent e1{false};
    AutoResetEvent e2{false};
    auto result = std::async([&e1, &e2] { return WaitAny({&e1, &e2}); });
    e2.Set();
    ASSERT_EQ(1, result.get());
}