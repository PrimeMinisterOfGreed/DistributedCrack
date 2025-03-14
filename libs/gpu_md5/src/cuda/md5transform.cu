#include "md5gpu.cuh"
#include "md5transform.cuh"
// Constants for MD5Transform routine.

__global__ void md5_apply_gpu(uint8_t* data, uint32_t* sizes, uint32_t * offsets, uint32_t* result, size_t numofstring){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if(i < numofstring){
    auto size = sizes[i];
    char* str = reinterpret_cast<char*>(&data[offsets[i]]);
    MD5Gpu alg{str,size};
    auto digest = alg.getdigest();
    memcpy(&result[i*4],digest,16);
  }
}





void md5_gpu_transform(uint8_t *data, uint32_t *sizes, uint32_t *result,
                       size_t num_of_strings, int maxthreads) {

  uint8_t *_devdata = nullptr;
  uint32_t *_devsizes = nullptr,*_devresult = nullptr,
      *offsets = new uint32_t[num_of_strings]{},
      *_devoffsets = nullptr;

  // CheckGpuCondition();
  size_t cumsizes = sizes[0];
  offsets[0] = 0;
  for (uint32_t i = 1; i < num_of_strings; i++) {
    offsets[i] = cumsizes;
    cumsizes += sizes[i];
  }
  cudaMalloc(&_devdata, cumsizes);
  cudaMalloc(&_devresult, 4 * num_of_strings*sizeof(uint32_t));
  cudaMalloc(&_devsizes, num_of_strings*sizeof(uint32_t));
  cudaMalloc(&_devoffsets, num_of_strings*sizeof(uint32_t));

  cudaMemcpy(_devdata, data, cumsizes, cudaMemcpyHostToDevice);
  cudaMemcpy(_devsizes, sizes, num_of_strings*sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(_devoffsets, offsets, num_of_strings*sizeof(uint32_t) , cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  md5_apply_gpu<<<min((size_t)maxthreads,num_of_strings),(num_of_strings/(size_t)maxthreads) + 1>>>(_devdata, _devsizes, _devoffsets, _devresult, num_of_strings);
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  if(err != cudaSuccess){
    //throw CudaMemoryError(err);
  }
  cudaMemcpy(result, _devresult, 4 * num_of_strings*sizeof(uint32_t), cudaMemcpyDeviceToHost);
  free(offsets);
  cudaFree(_devdata);
  cudaFree(_devresult);
  cudaFree(_devsizes);
  cudaFree(_devoffsets);  
}
