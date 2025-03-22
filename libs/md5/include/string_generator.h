#pragma once
#include <stddef.h>
#include <stdint.h>

const int minCharInt = 33;
const int maxCharint = 126;
const int minDigit = 48;
const int maxDigit = 57;
const int minUpperCaseLetter = 65;
const int maxUpperCaseLetter = 90;
const int minLowerCaseLetter = 97;
const int maxLowerCaseLetter = 122;


struct SequenceGeneratorCtx{
  char buffer[32];
  uint64_t index;
  uint8_t base_len;
  uint8_t current_len;
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