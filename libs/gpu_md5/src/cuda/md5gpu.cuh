#pragma once
#include "cuda.h"
#include "cuda_runtime.h"


constexpr const int blocksize = 64;
class MD5Gpu
{
  public:
    typedef unsigned int size_type; // must be 32bit

   __device__ MD5Gpu();
   __device__ MD5Gpu(const char * data, size_t size);
   __device__ void update(const unsigned char *buf, size_type length);
   __device__ MD5Gpu &finalize();
   __device__ const uint8_t* getdigest() const{return digest;}

  private:
   __device__  void init();
   __device__  void transform(const uint8_t block[blocksize]);
   __device__  static void decode(uint32_t output[], const uint8_t input[], size_type len);
   __device__  static void encode(uint8_t output[], const uint32_t input[], size_type len);
    bool finalized;
    uint8_t buffer[blocksize]{}; // bytes that didn't fit in last 64 byte chunk
    uint32_t count[2]{};          // 64bit counter for number of bits (lo, hi)
    uint32_t state[4]{};          // digest so far
    uint8_t digest[16]{};        // the result
};