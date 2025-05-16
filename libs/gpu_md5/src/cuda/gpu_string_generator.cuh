#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include "_cdecl" 
CDECL
struct GpuStringGenerator{
      uint8_t base_len;
      uint8_t current_len;
      size_t index;
      char buffer[32];
};
__device__ GpuStringGenerator new_generator(uint8_t initialSequenceLength);
__device__ void assign_address(GpuStringGenerator *gen, size_t address);
__device__ void generate_chunk(GpuStringGenerator *gen, size_t size,
                               char *data);
__device__ void next_sequence(GpuStringGenerator*self,char *data);
__device__ void destroy_generator(GpuStringGenerator *gen);
__device__ size_t current_len(GpuStringGenerator *gen);
END