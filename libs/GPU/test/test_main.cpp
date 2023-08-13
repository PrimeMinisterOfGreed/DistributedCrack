//
// Created by drfaust on 10/02/23.
//
#include <gtest/gtest.h>
#include <cuda.h>
#include <md5Cuda.cuh>
int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    CheckGpuCondition();
    return RUN_ALL_TESTS();
}


