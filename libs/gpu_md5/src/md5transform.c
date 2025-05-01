#include "md5_gpu.h"
#include "cuda/md5transform.cuh"

struct Md5TransformResult md5_gpu(char* data, uint8_t * sizes, size_t array_size, int maxthreads){
    struct Md5TransformResult result;
    struct md5_transform_request request = new_request(data, sizes, array_size);
    md5_gpu_transform(request, maxthreads);
    result.data = request.result;
    result.size = array_size;
    return result;
}

