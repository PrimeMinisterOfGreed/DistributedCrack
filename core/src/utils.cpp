#include "utils.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>


SequenceGenerator::SequenceGenerator(uint8_t base_len) {
    ctx = new_seq_generator(base_len);
}

size_t SequenceGenerator::absolute_index() const {
    return ctx.index + (span * (ctx.base_len - 1));
}

size_t SequenceGenerator::remaining_this_size() const {
    return 93 * (ctx.current_len + 1) - absolute_index();
}

void SequenceGenerator::next_sequence() {
    seq_gen_next_sequence(&ctx);
}

void SequenceGenerator::skip_to(size_t address) {
    seq_gen_skip_to(&ctx, address);
}

const char* SequenceGenerator::get_buffer() const {
    return ctx.buffer;
}

std::string SequenceGenerator::current() const {
    std::string s{ctx.buffer, ctx.current_len};
    return s;
}

GeneratorResult SequenceGenerator::generate_flatten_chunk(size_t chunks) {
    GeneratorResult result;
    result.strings.reserve((ctx.current_len + 1) * span);
    result.sizes.reserve(chunks);
    for (size_t i = 0; i < chunks; i+=1) {
        result.sizes.push_back(ctx.current_len);
        for (uint8_t j = 0; j < ctx.current_len; j+=1) {
            if (ctx.buffer[j] == 0) break;
            result.strings.push_back(static_cast<uint8_t>(ctx.buffer[j]));
        }
        next_sequence();
    }
    return result;
}

DictionaryReader::DictionaryReader(const char* filename)
{
    file = std::ifstream(filename, std::ios::in);
    if (!file.is_open()) {
        printf("Error opening file: %s\n", filename);
        abort();
    }   

}

GeneratorResult DictionaryReader::generate_flatten_chunk(size_t chunks) {
    GeneratorResult result;
    result.strings.reserve(chunks * 8);
    result.sizes.reserve(chunks);
    for (size_t i = 0; i < chunks; i+=1) {
        std::string line;
        std::getline(file, line);
        if (file.eof()) {
            break;
        }
        result.sizes.push_back(line.size());
        result.strings.insert(result.strings.end(), line.begin(), line.end());
    }
    return result;
}
