#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include "string_generator.h"
#include <fstream>


struct GeneratorResult {
    std::vector<uint8_t> strings;
    std::vector<uint8_t> sizes;
};


struct ChunkGenerator{
    virtual GeneratorResult generate_flatten_chunk(size_t chunks) = 0;   
};

struct SequenceGenerator : ChunkGenerator {
public:
    SequenceGenerator(uint8_t base_len);
    size_t absolute_index() const;
    size_t remaining_this_size() const;
    void next_sequence();
    void skip_to(size_t address);
    const char* get_buffer() const;
    std::string current() const;
    size_t size() const { return ctx.current_len; }
    virtual GeneratorResult generate_flatten_chunk(size_t chunks) override;
private:
    SequenceGeneratorCtx ctx;
    static constexpr size_t span = 94;
};



struct DictionaryReader : public ChunkGenerator{
    private:
    std::ifstream file;
    public:
    DictionaryReader(const char* filename);
    virtual GeneratorResult generate_flatten_chunk(size_t chunks) override;
};