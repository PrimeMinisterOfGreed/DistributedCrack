/* MD5
 converted to C++ class by Frank Thilo (thilo@unix-ag.org)
 for bzflag (http://www.bzflag.org)

   based on:

   md5.h and md5.c
   reference implementation of RFC 1321

   Copyright (C) 1991-2, RSA Data Security, Inc. Created 1991. All
rights reserved.

License to copy and use this software is granted provided that it
is identified as the "RSA Data Security, Inc. MD5 Message-Digest
Algorithm" in all material mentioning or referencing this software
or this function.

License is also granted to make and use derivative works provided
that such works are identified as "derived from the RSA Data
Security, Inc. MD5 Message-Digest Algorithm" in all material
mentioning or referencing the derived work.

RSA Data Security, Inc. makes no representations concerning either
the merchantability of this software or the suitability of this
software for any particular purpose. It is provided "as is"
without express or implied warranty of any kind.

These notices must be retained in any copies of any part of this
documentation and/or software.

*/

#pragma once

#include <cstring>
#include <iostream>
#include <cstddef>
#include <cstdint>
// a small class for calculating MD5 hashes of strings or byte arrays
// it is not meant to be fast or secure
//
// usage: 1) feed it blocks of uchars with update()
//      2) finalize()
//      3) get hexdigest() string
//      or
//      MD5(std::string).hexdigest()
//
// assumes that char is 8 bit and int is 32 bit
class MD5
{
  public:
    typedef unsigned int size_type; // must be 32bit

    MD5();
    MD5(const std::string &text);
    void update(const unsigned char *buf, size_type length);
    void update(const char *buf, size_type length);
    MD5 &finalize();
    std::string hexdigest() const;
    friend std::ostream &operator<<(std::ostream &, MD5 md5);

  private:
    void init();
    enum
    {
        blocksize = 64
    }; // VC6 won't eat a const static int here

    void transform(const uint8_t block[blocksize]);
    static void decode(uint32_t output[], const uint8_t input[], size_type len);
    static void encode(uint8_t output[], const uint32_t input[], size_type len);

    bool finalized;
    uint8_t buffer[blocksize]{}; // bytes that didn't fit in last 64 byte chunk
    uint32_t count[2]{};          // 64bit counter for number of bits (lo, hi)
    uint32_t state[4]{};          // digest so far
    uint8_t digest[16]{};        // the result

};

std::string md5(const std::string str);