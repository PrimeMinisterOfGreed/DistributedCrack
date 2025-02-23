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

struct GpuStringGenerator
{
      int _initialSequenceLength = 0;
      size_t _currentAddress = 0;
      size_t _currentSequenceLength = 0;
      char * _current;
      bool _current_used= false;
     __device__ GpuStringGenerator(int initialSequenceLength);

     __device__ void assign_address(size_t address);
     __device__ void generate_chunk(size_t size, char* data);
     __device__ void next_sequence(char* data);
    };

