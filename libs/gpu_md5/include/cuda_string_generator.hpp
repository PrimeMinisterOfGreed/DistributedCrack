#pragma once
#include <string>
#include<vector>


class mmp_string_generator{

    public:
    virtual std::vector<std::string> generate_chunk(size_t size) = 0;
    virtual void assign_address(size_t address) = 0;
};


mmp_string_generator* new_gpu_generator(int device);
