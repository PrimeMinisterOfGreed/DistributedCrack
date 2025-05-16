#include <cmath>
#include <gtest/gtest.h>
extern "C" {
#include "md5.h"
#include "string_generator.h"
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

TEST(TestGenerator, test_known_sequence) {
  struct SequenceGeneratorCtx ctx = new_seq_generator(4);
  seq_gen_skip_to(&ctx, 372);
  char *seq = ctx.buffer;
  ASSERT_EQ(seq[ctx.current_len - 1], (char)minCharInt);
  ASSERT_EQ(seq[ctx.current_len - 2], (char)minCharInt + 4);
}

TEST(TestGenerator, test_sequence_increment) {
  for (int i = 1; i < 32; i++) {
    struct SequenceGeneratorCtx generator = new_seq_generator(1);
    seq_gen_skip_to(&generator, pow(94, i));
    char *seq = generator.buffer;
    for(int j = 0; j < i; j++) {
      ASSERT_EQ(seq[j], (char)minCharInt) << "At index" << j << " of sequence: " << seq << " with length: " << i;
    }
    ASSERT_EQ(generator.current_len, i+1) << "Current sequence: " << seq;
  }
}

TEST(TestGenerator, test_sequence_increment_2) {
  for (int i = 2; i < 32; i++) {
    struct SequenceGeneratorCtx generator = new_seq_generator(1);
    seq_gen_skip_to(&generator, pow(94, i)-1);
    char *seq = generator.buffer;
    for(int j = 0; j < i - 1; j++) {
      ASSERT_EQ(seq[j], (char)maxCharint) << "At index" << j << " of sequence: " << seq << " with length: " << i;
    }
    ASSERT_EQ(generator.current_len, i) << "Current sequence: " << seq;
  }
}

TEST(testMd5, test_md5) {
  struct MD5Context ctx;
  char buffer[16];
  memset(buffer, 0, 16);
  char result[32];
  md5String((char *)"0000", (uint8_t *)buffer);
  md5HexDigest((uint8_t *)buffer, result);
  printf("computed: %s\n", result);
  printf("original %s\n", "4a7d1ed414474e4033ac29ccb8653d9b");
  ASSERT_TRUE(strcmp(result, "4a7d1ed414474e4033ac29ccb8653d9b") == 0);
}


TEST(TestGenerator, test_word_generation){
  std::vector<std::string>  words {"hello", "world", "test", "string"};
  struct SequenceGeneratorCtx ctx = new_seq_generator(4);
  int old_len = ctx.base_len;
  for(size_t i = 0;; i++){
    seq_gen_skip_to(&ctx, i);

  }

}