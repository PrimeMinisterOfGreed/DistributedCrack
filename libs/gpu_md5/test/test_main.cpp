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

TEST(TestGpu, test_simple_calc){
    std::string tgt = "5eb63bbbe01eeed093cb22bb8f5acdc3";
    auto index = md5_gpu({"hello world"});
    ASSERT_EQ(index[0],tgt);
}

TEST(TestGpu, test_md5_calc) {
    std::vector<std::string> chunk{"foo","bar","hello world"};
    std::string tgt = "5eb63bbbe01eeed093cb22bb8f5acdc3";
    auto index = md5_gpu(chunk);
    ASSERT_EQ(index[2],tgt);
}

TEST(TestGpu, test_regime_response){
  AssignedSequenceGenerator generator{4};
  auto chunk = generator.generate_chunk(100000);
  auto digests = std::vector<std::string>{chunk.size()};
  #pragma omp parallel
  for(auto i = 0 ; i < chunk.size(); i++){
    digests[i] = md5(chunk[i]);
  }
  auto gpucomputed = md5_gpu(chunk);
  for(auto i = 0 ; i < chunk.size(); i++){
    ASSERT_EQ(gpucomputed[i], digests[i]);
  }

}

bool target_found(std::string& target, std::vector<std::string>& result){
  #pragma omp parallel for threads(16)
    for(int i = 0 ; i < result.size() ; i++){
      if(result[i] == target) return true;
    }
  
  return false;
}

TEST(BenchMark, gpu_benchmark){
  AssignedSequenceGenerator generator{4};
  std::vector<std::string> res{};
  auto target_md5 =  std::string("df0ab011de60a20038a6d5fd760de52e");
  auto chunklevel = 1000000;
  do{
    auto chunk = generator.generate_chunk(chunklevel);
    res = md5_gpu(chunk,chunklevel);
    chunklevel += 100000;
    printf("current chunk level %d \n",chunklevel);
    fflush(NULL);
  } while(!target_found(target_md5, res)); 
}

TEST(TestGpu, test_bruter){
  std::string target = "98abe3a28383501f4bfd2d9077820f11";
  std::string orig = "!!!!";
  auto res = md5_bruter(0, 10, target,10,4);
  ASSERT_TRUE(res.has_value());
  ASSERT_EQ(res.value(), orig);
}