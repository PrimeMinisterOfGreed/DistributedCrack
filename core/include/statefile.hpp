#pragma once 
#include <cstddef>
#include <memory>
#include <optional>
struct StateFile 
{
    size_t current_address = 0;
    char current_dictionary[256]{};
    char filename[256]{};
    static std::optional<std::unique_ptr<StateFile>> load(const char* filename);
    static std::unique_ptr<StateFile> create(const char* filename);
    void save() const;
};