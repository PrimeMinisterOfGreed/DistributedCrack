#pragma once 
#include <cstddef>
#include <memory>
struct StateFile 
{
    size_t current_address = 0;
    char current_dictionary[256]{};

    static std::unique_ptr<StateFile> load(const char* filename);
    void save(const char* filename) const;
};