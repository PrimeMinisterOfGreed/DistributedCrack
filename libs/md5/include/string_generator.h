#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

const int minCharInt = 33;
const int maxCharint = 126;



struct SequenceGeneratorCtx{
  uint64_t index;
  uint8_t current_len;
  uint8_t base_len;
  char buffer[32]; // placed at the end, so that it can be extended
};

struct SequenceIterator{
  char * buffer;
  uint8_t * sizes;
  uint32_t size;
  uint32_t index;
};

struct SequenceGeneratorCtx new_seq_generator(uint8_t base_len);
void seq_gen_next_sequence(struct SequenceGeneratorCtx* ctx);
void seq_gen_skip_to(struct SequenceGeneratorCtx* ctx,size_t address);

#ifdef __cplusplus
}
#endif