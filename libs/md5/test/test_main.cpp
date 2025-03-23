#include <gtest/gtest.h>
extern "C"{
#include "md5.h"
#include "string_generator.h"
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

TEST(TestGenerator, test_known_sequence) {
  struct SequenceGeneratorCtx ctx = new_seq_generator(4);
  seq_gen_skip_to(&ctx,372);
  char* seq = ctx.buffer;
  ASSERT_EQ(seq[ctx.current_len - 1], (char)minCharInt);
  ASSERT_EQ(seq[ctx.current_len - 2] ,(char)minCharInt + 4);
}

TEST(TestGenerator, test_sequence_increment) {
  struct SequenceGeneratorCtx generator = new_seq_generator(1);
  seq_gen_skip_to(&generator, 93);
  seq_gen_next_sequence(&generator);
  char* seq = generator.buffer;
  ASSERT_EQ(seq[0] , (char)minCharInt);
  ASSERT_EQ(seq[1] , (char)minCharInt + 1);
  ASSERT_EQ(generator.current_len, 2);
}

TEST(TestGenerator, test_sequence_increment2) {
  struct SequenceGeneratorCtx generator = new_seq_generator(1);
  seq_gen_skip_to(&generator, 186);
  seq_gen_next_sequence(&generator);
  char* seq = generator.buffer;
  ASSERT_EQ(seq[generator.current_len -1] , (char)minCharInt);
  ASSERT_EQ(seq[generator.current_len - 2] , (char)minCharInt + 1);
  ASSERT_EQ(generator.current_len, 3);
}


TEST(testMd5, test_md5) {
  struct MD5Context ctx;
  char buffer[16];
  memset(buffer, 0, 16);
  char result [32];
  md5String((char*)"0000",(uint8_t*) buffer);
  md5HexDigest((uint8_t*)buffer, result);
  printf("computed: %s\n", result);
  printf("original %s\n", "4a7d1ed414474e4033ac29ccb8653d9b");
  ASSERT_TRUE( strcmp(result,"4a7d1ed414474e4033ac29ccb8653d9b") == 0);
}

