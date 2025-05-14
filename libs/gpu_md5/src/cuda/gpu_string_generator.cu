#include "gpu_string_generator.cuh"
#include "_cdecl"
 

constexpr int minCharInt = 33;
constexpr int maxCharint = 126;




static inline __device__ void shift_buffer(char* buffer, int len, int shift)
{
    memcpy(buffer+shift, buffer, len);
}

__device__ void assign_address(GpuStringGenerator* ctx,size_t address)
{
    int div = maxCharint - minCharInt;
    int q = address;
    int r = 0;
    int it = 0;
    while (q > 0)
    {
        r = q % div;
        q /= div;
        if (it == ctx->current_len)
        {
            shift_buffer(ctx->buffer, ctx->current_len, 1); 
            ctx->current_len++;
            ctx->buffer[0] = minCharInt;
        }
        ctx->buffer[ctx->current_len - it - 1] = (char)(r + minCharInt);
        it++;
    }
    ctx->index =address;
}

__device__ GpuStringGenerator new_generator(uint8_t base_len)
{
    GpuStringGenerator gen{};
    memset(&gen,0,sizeof(GpuStringGenerator));
    gen.base_len = base_len;
    gen.current_len = base_len;
    for (int i = 0; i < (base_len>0?base_len:1); i++)
    {
        gen.buffer[i] += minCharInt;
    }
    return gen;
}


__device__ void next_sequence(GpuStringGenerator* ctx,char* data){
for (int i = ctx->current_len - 1; i >= 0; i--)
    {
        ctx->buffer[i]++;
        if (ctx->buffer[i] > maxCharint)
        {
            ctx->buffer[i] = minCharInt;
            if (i == 0)
            {
                shift_buffer(ctx->buffer, ctx->current_len, 1);
                ctx->current_len++;
                ctx->buffer[0] = minCharInt;
            }
        }
        else
        {
            break;
        }
    }
}

__device__ void destroy_generator(GpuStringGenerator* self){
    
}

