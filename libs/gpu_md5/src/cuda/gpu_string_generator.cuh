#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include "_cdecl" 
CDECL
struct GpuStringGenerator{
      char current[24];
      int initialSequenceLength;
      int currentSequenceLength;
      bool current_used;
};
__device__ GpuStringGenerator new_generator(int initialSequenceLength);
__device__ void assign_address(GpuStringGenerator *gen, size_t address);
__device__ void generate_chunk(GpuStringGenerator *gen, size_t size,
                               char *data);
__device__ void next_sequence(GpuStringGenerator*self,char *data);
__device__ void destroy_generator(GpuStringGenerator *gen);
END