#include "StatusSaver.hpp"
#include <boost/serialization/access.hpp>
#include <boost/serialization/serialization.hpp>
#include <gtest/gtest.h>

struct MockStruct
{
    friend class boost::serialization::access;
    int a;
    int b;

    template <class Archive> void serialize(Archive &ar, const unsigned int version)
    {
        ar &a;
        ar &b;
    }
};

TEST(TestSave, test_save_struct)
{
    BaseStatusSaver<MockStruct> saver{"test.archive"};
    saver.Save(*new MockStruct{1, 2});
    auto loaded = saver.Restore();
    ASSERT_EQ(1, loaded.a);
    ASSERT_EQ(2, loaded.b);
}