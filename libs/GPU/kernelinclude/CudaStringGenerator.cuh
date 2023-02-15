#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

constexpr int minCharInt = 33;
constexpr int maxCharint = 126;
constexpr int minDigit = 48;
constexpr int maxDigit = 57;
constexpr int minUpperCaseLetter = 65;
constexpr int maxUpperCaseLetter = 90;
constexpr int minLowerCaseLetter = 97;
constexpr int maxLowerCaseLetter = 122;

class CudaStringGenerator
{
      private:
      int _initialSequenceLength = 0;
      size_t _currentAddress = 0;
      size_t _currentSequenceLength = 0;
      char * _current;
      public:
     __host__  __device__ const char * nextSequence();
     __host__  __device__ const char ** generateChunk(int num, size_t * sizes);
     __host__  __device__ void assignAddress(size_t address);
      CudaStringGenerator(int initialSequenceLength);
};

