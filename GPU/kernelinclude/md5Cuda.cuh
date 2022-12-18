#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <functional>

class MD5
{
  public:
    typedef unsigned int size_type; // must be 32bit

    __device__ __host__ MD5();
    __device__ __host__ MD5(const uint8_t *text, size_t size);
    __device__ __host__ void update(const unsigned char *buf, size_type length);
    __device__ __host__ void update(const char *buf, size_type length);
    __device__ __host__ MD5 &finalize();
    __device__ __host__ void exportdigest(uint8_t *dataOut);

  private:
    __device__ __host__ void init();
    typedef unsigned char uint1; //  8bit
    typedef unsigned int uint4;  // 32bit
    enum
    {
        blocksize = 64
    }; // VC6 won't eat a const static int here

    __device__ __host__ void transform(const uint1 block[blocksize]);
    __device__ __host__ static void decode(uint4 output[], const uint1 input[], size_type len);
    __device__ __host__ static void encode(uint1 output[], const uint4 input[], size_type len);

    bool finalized;
    uint1 buffer[blocksize]; // bytes that didn't fit in last 64 byte chunk
    uint4 count[2];          // 64bit counter for number of bits (lo, hi)
    uint4 state[4];          // digest so far
    uint1 digest[16];        // the result

    // low level logic operations
    __device__ __host__ static inline uint4 F(uint4 x, uint4 y, uint4 z);
    __device__ __host__ static inline uint4 G(uint4 x, uint4 y, uint4 z);
    __device__ __host__ static inline uint4 H(uint4 x, uint4 y, uint4 z);
    __device__ __host__ static inline uint4 I(uint4 x, uint4 y, uint4 z);
    __device__ __host__ static inline uint4 rotate_left(uint4 x, int n);
    __device__ __host__ static inline void FF(uint4 &a, uint4 b, uint4 c, uint4 d, uint4 x, uint4 s, uint4 ac);
    __device__ __host__ static inline void GG(uint4 &a, uint4 b, uint4 c, uint4 d, uint4 x, uint4 s, uint4 ac);
    __device__ __host__ static inline void HH(uint4 &a, uint4 b, uint4 c, uint4 d, uint4 x, uint4 s, uint4 ac);
    __device__ __host__ static inline void II(uint4 &a, uint4 b, uint4 c, uint4 d, uint4 x, uint4 s, uint4 ac);
};

__host__ __device__ void md5(const uint8_t *data, uint32_t size, uint8_t *result);
__host__ void md5_gpu(const uint8_t **data, const uint32_t *sizes, uint8_t **result, uint32_t size, int threads);

template <typename T> cudaError_t GpuMalloc(T **pointer, size_t size)
{
    return cudaMalloc(pointer, sizeof(T) * size);
}

template <typename T> cudaError_t GpuCopy(T *dst, T *src, size_t size, cudaMemcpyKind kind)
{
    return cudaMemcpy(dst, src, sizeof(T) * size, kind);
}

template <typename T> cudaError_t GpuManagedMalloc(T **pointer, size_t size)
{
    return cudaMallocManaged(pointer, sizeof(T) * size);
}