#include "cuda_memory_support.cuh"
#include "stdio.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Constants for MD5Transform routine.

constexpr uint8_t block_size = 64;


__device__ const uint32_t md5_magic_const[4] =
{   
 0x67452301,
 0xefcdab89,
 0x98badcfe,
 0x10325476,
};




constexpr uint8_t S11= 7;
constexpr uint8_t S12= 12;
constexpr uint8_t S13= 17;
constexpr uint8_t S14= 22;
constexpr uint8_t S21= 5;
constexpr uint8_t S22= 9;
constexpr uint8_t S23= 14;
constexpr uint8_t S24= 20;
constexpr uint8_t S31= 4;
constexpr uint8_t S32= 11;
constexpr uint8_t S33 = 16;
constexpr uint8_t S34 = 23;
constexpr uint8_t S41 = 6;
constexpr uint8_t S42 = 10;
constexpr uint8_t S43 = 15;
constexpr uint8_t S44 = 21;



///////////////////////////////////////////////

// F, G, H and I are basic MD5 functions.
__device__ constexpr inline uint32_t F(uint32_t x, uint32_t y, uint32_t z) {
  return x & y | ~x & z;
}

__device__ constexpr inline uint32_t G(uint32_t x, uint32_t y, uint32_t z) {
  return x & z | y & ~z;
}

__device__ constexpr inline uint32_t H(uint32_t x, uint32_t y, uint32_t z) {
  return x ^ y ^ z;
}

__device__ constexpr inline uint32_t I(uint32_t x, uint32_t y, uint32_t z) {
  return y ^ (x | ~z);
}

// rotate_left rotates x left n bits.
__device__ constexpr inline uint32_t rotate_left(uint32_t x, int n) {
  return (x << n) | (x >> (32 - n));
}

// FF, GG, HH, and II transformations for rounds 1, 2, 3, and 4.
// Rotation is separate from addition to prevent recomputation.
__device__ constexpr inline void FF(uint32_t &a, uint32_t b, uint32_t c, uint32_t d,
                         uint32_t x, uint32_t s, uint32_t ac) {
  a = rotate_left(a + F(b, c, d) + x + ac, s) + b;
}

__device__ constexpr inline void GG(uint32_t &a, uint32_t b, uint32_t c, uint32_t d,
                         uint32_t x, uint32_t s, uint32_t ac) {
  a = rotate_left(a + G(b, c, d) + x + ac, s) + b;
}

__device__ constexpr inline void HH(uint32_t &a, uint32_t b, uint32_t c, uint32_t d,
                         uint32_t x, uint32_t s, uint32_t ac) {
  a = rotate_left(a + H(b, c, d) + x + ac, s) + b;
}

__device__ constexpr inline void II(uint32_t &a, uint32_t b, uint32_t c, uint32_t d,
                         uint32_t x, uint32_t s, uint32_t ac) {
  a = rotate_left(a + I(b, c, d) + x + ac, s) + b;
}



__device__ constexpr uint8_t padding[block_size] = {0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                  0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                  0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

__device__  uint32_t byteswap(uint32_t word)
{
    return ((word >> 24) & 0x000000FF) | ((word >> 8) & 0x0000FF00) | ((word << 8) & 0x00FF0000) |
           ((word << 24) & 0xFF000000);
}

__device__ void transform(uint32_t state[4], const uint8_t block[block_size])
{
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t x[16];

    for (uint32_t i = 0, j = 0; j < block_size && i < 16; i++, j += 4)
    {
        x[i] = (uint)block[j] | ((uint)block[j + 1] << 8) | ((uint)block[j + 2] << 16) | ((uint)block[j + 3] << 24);
    }

    FF(a, b, c, d, x[0], S11, 0xd76aa478);
    FF(d, a, b, c, x[1], S12, 0xe8c7b756);
    FF(c, d, a, b, x[2], S13, 0x242070db);
    FF(b, c, d, a, x[3], S14, 0xc1bdceee);
    FF(a, b, c, d, x[4], S11, 0xf57c0faf);
    FF(d, a, b, c, x[5], S12, 0x4787c62a);
    FF(c, d, a, b, x[6], S13, 0xa8304613);
    FF(b, c, d, a, x[7], S14, 0xfd469501);
    FF(a, b, c, d, x[8], S11, 0x698098d8);
    FF(d, a, b, c, x[9], S12, 0x8b44f7af);
    FF(c, d, a, b, x[10], S13, 0xffff5bb1);
    FF(b, c, d, a, x[11], S14, 0x895cd7be);
    FF(a, b, c, d, x[12], S11, 0x6b901122);
    FF(d, a, b, c, x[13], S12, 0xfd987193);
    FF(c, d, a, b, x[14], S13, 0xa679438e);
    FF(b, c, d, a, x[15], S14, 0x49b40821);

    GG(a, b, c, d, x[1], S21, 0xf61e2562);
    GG(d, a, b, c, x[6], S22, 0xc040b340);
    GG(c, d, a, b, x[11], S23, 0x265e5a51);
    GG(b, c, d, a, x[0], S24, 0xe9b6c7aa);
    GG(a, b, c, d, x[5], S21, 0xd62f105d);
    GG(d, a, b, c, x[10], S22, 0x2441453);
    GG(c, d, a, b, x[15], S23, 0xd8a1e681);
    GG(b, c, d, a, x[4], S24, 0xe7d3fbc8);
    GG(a, b, c, d, x[9], S21, 0x21e1cde6);
    GG(d, a, b, c, x[14], S22, 0xc33707d6);
    GG(c, d, a, b, x[3], S23, 0xf4d50d87);
    GG(b, c, d, a, x[8], S24, 0x455a14ed);
    GG(a, b, c, d, x[13], S21, 0xa9e3e905);
    GG(d, a, b, c, x[2], S22, 0xfcefa3f8);
    GG(c, d, a, b, x[7], S23, 0x676f02d9);
    GG(b, c, d, a, x[12], S24, 0x8d2a4c8a);

    HH(a, b, c, d, x[5], S31, 0xfffa3942);
    HH(d, a, b, c, x[8], S32, 0x8771f681);
    HH(c, d, a, b, x[11], S33, 0x6d9d6122);
    HH(b, c, d, a, x[14], S34, 0xfde5380c);
    HH(a, b, c, d, x[1], S31, 0xa4beea44);
    HH(d, a, b, c, x[4], S32, 0x4bdecfa9);
    HH(c, d, a, b, x[7], S33, 0xf6bb4b60);
    HH(b, c, d, a, x[10], S34, 0xbebfbc70);
    HH(a, b, c, d, x[13], S31, 0x289b7ec6);
    HH(d, a, b, c, x[0], S32, 0xeaa127fa);
    HH(c, d, a, b, x[3], S33, 0xd4ef3085);
    HH(b, c, d, a, x[6], S34, 0x4881d05);
    HH(a, b, c, d, x[9], S31, 0xd9d4d039);
    HH(d, a, b, c, x[12], S32, 0xe6db99e5);
    HH(c, d, a, b, x[15], S33, 0x1fa27cf8);
    HH(b, c, d, a, x[2], S34, 0xc4ac5665);

    II(a, b, c, d, x[0], S41, 0xf4292244);
    II(d, a, b, c, x[7], S42, 0x432aff97);
    II(c, d, a, b, x[14], S43, 0xab9423a7);
    II(b, c, d, a, x[5], S44, 0xfc93a039);
    II(a, b, c, d, x[12], S41, 0x655b59c3);
    II(d, a, b, c, x[3], S42, 0x8f0ccc92);
    II(c, d, a, b, x[10], S43, 0xffeff47d);
    II(b, c, d, a, x[1], S44, 0x85845dd1);
    II(a, b, c, d, x[8], S41, 0x6fa87e4f);
    II(d, a, b, c, x[15], S42, 0xfe2ce6e0);
    II(c, d, a, b, x[6], S43, 0xa3014314);
    II(b, c, d, a, x[13], S44, 0x4e0811a1);
    II(a, b, c, d, x[4], S41, 0xf7537e82);
    II(d, a, b, c, x[11], S42, 0xbd3af235);
    II(c, d, a, b, x[2], S43, 0x2ad7d2bb);
    II(b, c, d, a, x[9], S44, 0xeb86d391);

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
}

__device__  void md5(const uint8_t *data, const uint32_t size, uint8_t *result)
{

    uint32_t i = 0;
    uint32_t state[4];
    memcpy(state,md5_magic_const,sizeof(state));
    for (i = 0; i + block_size <= size; i += block_size)
    {
        transform(state, data + i);
    }

    uint32_t size_in_bits = size << 3;
    uint8_t buffer[block_size];

    memcpy(buffer, data + i, size - i);
    memcpy(buffer + size - i, padding, block_size - (size - i));
    memcpy(buffer + block_size - (2 * sizeof(uint)), &size_in_bits, sizeof(uint));

    transform(state, buffer);

    memcpy(result, state, 4 * sizeof(uint));
}

__global__ void md5_gpu_comparer(const uint8_t *digests, const uint8_t *targetDigest, uint32_t size, uint32_t *result)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
    {
        uint8_t eq = 1;
        for (int j = 0; j < 16; j++)
        {
            if (digests[j + i * 16] != targetDigest[j])
            {
                eq = 0;
                break;
            }
        }
        if (eq)
            *result = i+1;
    }
}

__global__ void md5_call_gpu(const uint8_t *data, const uint32_t *sizes, uint32_t *offsets, uint8_t *result,
                             uint32_t size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
        md5(data + offsets[i], sizes[i], result + i * sizeof(uint32_t) * 4);
}

__host__ void select_cuda_device(){
    int devices;
    cudaGetDeviceCount(&devices);
    size_t maxFreeMem = 0;
    int betterDevice = 0;
    for(int i= 0; i < devices; i++)
    {
        cudaSetDevice(i);
        size_t freeMem;
        size_t totalMem;
        cudaMemGetInfo(&freeMem, &totalMem);
        if (freeMem > maxFreeMem){
            maxFreeMem= freeMem;
            betterDevice = i;
        }
    }
    cudaSetDevice(betterDevice);
    cudaDeviceReset();
}

struct _gpudata {
  uint8_t *_data = nullptr;
  uint32_t _size = 0;
  uint8_t *_result = nullptr;
  uint32_t *_displacements = nullptr;
  uint32_t *_sizes = nullptr;
  uint8_t *_target = nullptr;
  _gpudata(size_t num_of_strings, const uint8_t *local_data,
           const uint32_t *local_sizes, uint8_t *local_target) {
    uint32_t data_size = 0;
    for (size_t i = 0; i < num_of_strings; i++) {
      data_size += local_sizes[i];
    }
    uint32_t *local_displacements =
        (uint32_t *)alloca(num_of_strings * sizeof(uint32_t));
    memset(local_displacements, 0, num_of_strings * sizeof(uint32_t));
    for (size_t i = 1; i < num_of_strings; i++) {
      local_displacements[i] = local_sizes[i - 1] + local_displacements[i - 1];
    }
    alloc_gpu(num_of_strings, data_size, local_data, local_sizes,
              local_displacements, local_target);
  }
  void alloc_gpu(size_t num_of_string, size_t data_size,
                 const uint8_t *local_data, const uint32_t *local_sizes,
                 uint32_t *displacements, uint8_t *local_target) {
    _size = num_of_string;

        GpuMalloc(&_data, data_size);
        GpuMalloc(&_sizes, num_of_string);
        GpuMalloc(&_displacements, num_of_string);
        GpuCopy(_data, local_data, data_size, cudaMemcpyHostToDevice);
        GpuCopy(_sizes, local_sizes, num_of_string, cudaMemcpyHostToDevice);
        GpuCopy(_displacements, displacements, num_of_string,
                cudaMemcpyHostToDevice);
    GpuMalloc(&_result, num_of_string * 16);
    if (local_target != nullptr) {
      GpuMalloc(&_target, 16);
      GpuCopy(_target, local_target, 16, cudaMemcpyHostToDevice);
    }
  }

  void dealloc() {
    GpuFree(_data,_sizes,_displacements,_result);
    if (_target != NULL)
     GpuFree(_target);
  }
  ~_gpudata() { dealloc(); }
};

__host__ void CheckGpuCondition() {
  static bool initialized = false;
  CUresult result;
  if (!initialized && (result = cuInit(0)) != CUDA_SUCCESS)
    printf("Error %d on gpu initialization: %s\n", result,
           cudaGetErrorString((cudaError)result));  
  select_cuda_device();
}

__host__ int md5_gpu_finder(const uint8_t *data, const uint32_t *sizes,
                            uint32_t num_of_strings, uint8_t *targetDigest) {
  CheckGpuCondition();
  dim3 grid(1, num_of_strings);
  _gpudata gpudata{num_of_strings, data, sizes, targetDigest};
  md5_call_gpu<<<grid.x, grid.y>>>(gpudata._data, gpudata._sizes,
                                   gpudata._displacements, gpudata._result,
                                   num_of_strings);
  cudaDeviceSynchronize();

  md5_gpu_comparer<<<grid.x, grid.y>>>(gpudata._result, gpudata._target,
                                       num_of_strings, gpudata._displacements);

 cudaDeviceSynchronize();
  return 0;
}

__host__ void md5_gpu_transform(const uint8_t *data, const uint32_t *sizes,
                                uint8_t *result, uint32_t num_of_strings) {
  CheckGpuCondition();
  _gpudata gpudata{num_of_strings, data, sizes, NULL};
  cudaDeviceSynchronize();
  md5_call_gpu<<<1, num_of_strings>>>(gpudata._data, gpudata._sizes,
                                      gpudata._displacements, gpudata._result,
                                      num_of_strings);

  cudaDeviceSynchronize();
  GpuCopy(result, gpudata._result, 16 * num_of_strings, cudaMemcpyDeviceToHost);

}


