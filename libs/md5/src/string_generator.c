#include "string_generator.h"
#include <stdlib.h>
#include <string.h>



static inline void shift_buffer(char* buffer, int len, int shift)
{
    memcpy(buffer+shift, buffer, len);
}

void sequence_generator_init(struct SequenceGeneratorCtx* ctx, uint8_t base_len)
{
    memset(ctx->buffer, 0, sizeof(ctx->buffer));
    for (int i = 0; i < (base_len>0?base_len:1); i++)
    {
        ctx->buffer[i] += minCharInt;
    }
}

void seq_gen_next_sequence(struct SequenceGeneratorCtx* ctx)
{
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
};







void seq_gen_skip_to(struct SequenceGeneratorCtx* ctx,uint64_t address)
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
}



struct SequenceGeneratorCtx new_seq_generator( uint8_t base_len)
{
    struct SequenceGeneratorCtx result;
    result.base_len = base_len;
    result.current_len = base_len;
    result.index = 0;
    sequence_generator_init(&result, base_len);

    return result;  
}

