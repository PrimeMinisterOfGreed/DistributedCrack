#include "md5.cuh"
#include "md5transform.cuh"
#include "_cdecl"
CDECL
// Constants for MD5Transform routine.

struct exception{
  cudaError_t error;
  int line;
  const char* file;
};
#define dbgline printf("process %d line reached %d\n",i,__LINE__);
#define try(expr) if((_exc.error = expr) != cudaSuccess){_exc.line = __LINE__; _exc.file = __FILE_NAME__ ; goto ERROR;}
#define decl_exc struct exception _exc;
#define handle ERROR: printf("Error: %s at %s:%d\n",cudaGetErrorString(_exc.error),_exc.file,_exc.line);

__host__ size_t static inline get_data_size(uint32_t *sizes, size_t num_of_strings){
  size_t cumsizes = sizes[0];
  for (uint32_t i = 1; i < num_of_strings; i++) {
    cumsizes += sizes[i];
  }
  return cumsizes;
}

__global__ void md5_apply_gpu(struct md5_transform_request req){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if(i < req.num_of_strings){
    size_t size = req.sizes[i];
    uint8_t *data = req.data + req.offsets[i];
    uint8_t digest[16];
    char result[33];
    memset(result,0,33);
    md5String((char*)data,digest,size);
    md5HexDigest(digest, result);
    memcpy(req.result + i*33,result,32);
    printf("result: %s\n",req.result + i*33);
  }
}

__host__ static inline struct md5_transform_request copy_trasform_request(struct md5_transform_request * req){
  decl_exc;
  struct md5_transform_request devptr;
  memset(&devptr,0,sizeof(struct md5_transform_request));
  devptr.num_of_strings = req->num_of_strings;
  size_t data_size = get_data_size(req->sizes, req->num_of_strings);
  try(cudaMalloc(&devptr.data, data_size));
  try(cudaMalloc(&devptr.sizes, req->num_of_strings*sizeof(uint32_t)));
  try(cudaMalloc(&devptr.result, req->num_of_strings*33));
  try(cudaMalloc(&devptr.offsets, req->num_of_strings*sizeof(uint32_t)));
  try(cudaMemcpy(devptr.data, req->data, data_size, cudaMemcpyHostToDevice));
  try(cudaMemcpy(devptr.sizes, req->sizes, req->num_of_strings*sizeof(uint32_t), cudaMemcpyHostToDevice));
  try(cudaMemcpy(devptr.offsets, req->offsets, req->num_of_strings*sizeof(uint32_t), cudaMemcpyHostToDevice));
  return devptr;
  handle;
  return devptr;
}

__host__ static inline void free_transform_request(struct md5_transform_request req){
  decl_exc;
  try(cudaFree(req.data));
  try(cudaFree(req.sizes));
  try(cudaFree(req.result));
  try(cudaFree(req.offsets));
  return;
  handle;
}

void md5_gpu_transform(struct md5_transform_request request, int maxthreads) {
  decl_exc;
  struct md5_transform_request devptr= copy_trasform_request(&request);
  cudaDeviceSynchronize();
  md5_apply_gpu<<<min((size_t)maxthreads,request.num_of_strings),(request.num_of_strings/(size_t)maxthreads) + 1>>>(devptr);
  cudaDeviceSynchronize();
  try(cudaGetLastError());
  cudaMemcpy(request.result, devptr.result, request.num_of_strings*33, cudaMemcpyDeviceToHost);
  free_transform_request(devptr);
  return;
  handle;
}


struct md5_transform_request new_request(uint8_t *data, uint32_t *sizes, size_t num_of_strings){
  struct md5_transform_request req;
  req.data = data;
  req.sizes = sizes;
  req.num_of_strings = num_of_strings;
  req.offsets = (uint32_t*)malloc(num_of_strings*sizeof(uint32_t));
  req.result = (char*)malloc(num_of_strings*33);
  req.offsets[0] = 0;
  for (uint32_t i = 1; i < num_of_strings; i++) {
    req.offsets[i] = req.offsets[i-1] + sizes[i-1];
  }
  return req;
}


END