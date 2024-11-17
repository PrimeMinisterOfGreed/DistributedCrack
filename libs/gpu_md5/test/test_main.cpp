//
// Created by drfaust on 10/02/23.
//
#include <gtest/gtest.h>
#include "md5_gpu.hpp"
int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

TEST(TestGpuResponse, test_md5_calc) {
    std::vector<std::string> chunk{"foo","bar","hello world"};
    std::string tgt = "5eb63bbbe01eeed093cb22bb8f5acdc3";
    auto index = md5_gpu(chunk,3,tgt);
    ASSERT_EQ(2, index);
    printf("Response from gpu %d",index);
}