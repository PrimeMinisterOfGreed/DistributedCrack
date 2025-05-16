#include "_cdecl"
#include "gpu_string_generator.cuh"

constexpr int minCharInt = 33;
constexpr int maxCharint = 126;

static __device__ void shift_buffer(char *buffer, int len, int shift) {
  char temp[32];
  memset(temp, 0, sizeof(temp));
  memset(temp, minCharInt, 32);
  memcpy(temp + shift, buffer, len);
  memcpy(buffer, temp, len+1);
}

__device__ void skip_to(GpuStringGenerator *ctx, size_t address) {
  int div = (maxCharint - minCharInt) + 1;
  size_t q = address;
  size_t r = 0;
  size_t it = 0;
  do {
    r = q % div;
    q /= div;
    if (it == ctx->current_len - 1 && q > 0) {
      shift_buffer(ctx->buffer, ctx->current_len, 1);
      ctx->current_len++;
    }

    ctx->buffer[ctx->current_len - it - 1] = (char)(r + minCharInt);
    it++;

  } while (q > 1);
}

__device__ GpuStringGenerator new_generator(uint8_t base_len) {
  GpuStringGenerator gen{};
  memset(&gen, 0, sizeof(GpuStringGenerator));
  gen.base_len = base_len;
  gen.current_len = base_len;
  memset(gen.buffer, minCharInt, base_len);
  return gen;
}

__device__ void next_sequence(GpuStringGenerator *ctx, char *data) {
  for (int i = ctx->current_len - 1; i >= 0; i--) {
    ctx->buffer[i]++;
    if (ctx->buffer[i] > maxCharint) {
      ctx->buffer[i] = minCharInt;
      if (i == 0) {
        shift_buffer(ctx->buffer, ctx->current_len, 1);
        ctx->current_len++;
        ctx->buffer[0] = minCharInt;
      }
    } else {
      break;
    }
  }
}

__device__ void destroy_generator(GpuStringGenerator *self) {}

__device__ size_t current_len(GpuStringGenerator *self) {
  return self->current_len;
}