#ifndef MD5_H
#define MD5_H

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif
struct MD5Context{
    uint64_t size;        // Size of input in bytes
    uint32_t buffer[4];   // Current accumulation of hash
    uint8_t input[64];    // Input to be used in the next step
    uint8_t digest[16];   // Result of algorithm
};

void md5Init(struct MD5Context *ctx);
void md5Update(struct MD5Context *ctx, uint8_t *input, size_t input_len);
void md5Finalize(struct MD5Context *ctx);
void md5Step(uint32_t *buffer, uint32_t *input);

void md5String(char *input, uint8_t *result);
void md5File(FILE *file, uint8_t *result);

void md5HexDigest(uint8_t *digest, char *result);

#endif
#ifdef __cplusplus
}
#endif