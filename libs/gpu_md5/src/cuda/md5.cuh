#pragma once
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "_cdecl"
CDECL
struct MD5Context{
    uint64_t size;        // Size of input in bytes
    uint32_t buffer[4];   // Current accumulation of hash
    uint8_t input[64];    // Input to be used in the next step
    uint8_t digest[16];   // Result of algorithm
};


__device__ void md5Init(struct MD5Context *ctx);
__device__ void md5Update(struct MD5Context *ctx, uint8_t *input, size_t input_len);
__device__ void md5Finalize(struct MD5Context *ctx);
__device__ void md5Step(uint32_t *buffer, uint32_t *input);
__device__ void md5String(char *input, uint8_t *result, size_t size);
__device__ void md5HexDigest(const uint8_t digest[16], char result[33]);
END