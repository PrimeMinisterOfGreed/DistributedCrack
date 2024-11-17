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

class MMPStringGenerator
{
      private:
      int _initialSequenceLength = 0;
      size_t _currentAddress = 0;
      size_t _currentSequenceLength = 0;
      char * _current;
     __device__ const char * next_sequence();

   public:
     MMPStringGenerator(int initialSequenceLength);

     __device__ void assign_address(size_t address, const char* seq, size_t* seqsize);
};

__global__ void generate_chunk(int num, size_t *sizes, const char **chunk);
