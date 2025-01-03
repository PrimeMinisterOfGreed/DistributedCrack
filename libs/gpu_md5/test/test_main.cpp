//
// Created by drfaust on 10/02/23.
//
#include <gtest/gtest.h>
#include "md5.hpp"
#include "md5_gpu.hpp"
#include "string_generator.hpp"
#include "cuda.h"
#include "cuda_runtime_api.h"

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  auto res = RUN_ALL_TESTS();
  cudaDeviceSynchronize();
  return res;
}

TEST(TestGpuResponse, test_md5_calc) {
    std::vector<std::string> chunk{"foo","bar","hello world"};
    std::string tgt = "5eb63bbbe01eeed093cb22bb8f5acdc3";
    auto index = md5_gpu(chunk);
    ASSERT_EQ(index[2],tgt);
}

TEST(TestGpuResponse, test_regime_response){
  AssignedSequenceGenerator generator{4};
  auto chunk = generator.generate_chunk(10000);
  auto digests = std::vector<std::string>{chunk.size()};
  #pragma omp parallel
  for(auto i = 0 ; i < chunk.size(); i++){
    digests[i] = md5(chunk[i]);
  }

  for(int i = 0; i < chunk.size(); i++){
    auto res = md5_gpu(chunk,digests[i]);
    ASSERT_EQ(res, i);
  } 
}