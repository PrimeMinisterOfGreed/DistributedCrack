#include "chunk_loader.hpp"
#include "log_engine.hpp"
#include "options_bag.hpp"
#include <cassert>
#include <cstring>
#include <filesystem>

static void operator <<(char buffer[], std::string value){
    memcpy(buffer,value.c_str(),value.size());
}


ChunkLoader::ChunkLoader()
{
    using namespace std::filesystem;
    _state_store.open("state.dat");
    char buffer[sizeof(LoaderState)]{};
    _state_store >> buffer;
    memcpy(&_state, buffer, sizeof(LoaderState));
    if (options.dictionary != _state.filename) {
      _state.filename << options.dictionary;
      _state.actualseq = 0;
    }
    if(options.use_dictionary()){
        assert(std::filesystem::exists(_state.filename));
        _dictionary.open(_state.filename);
    }
    else{
        _generator.assign_address(_state.actualseq);
    }
    _state_store.write(reinterpret_cast<char*>(&_state), sizeof(LoaderState));
}



std::vector<std::string> ChunkLoader::get_chunk(int dim) {
    std::vector<std::string> res{};
    if(options.use_dictionary()){
        std::string line{};
        for(int i = 0; i < dim; i++){
            if(!std::getline(_dictionary,line).eof() && line != "")
                res.push_back(line);
            else 
                break;
        }
    }
    else{
        res = _generator.generate_chunk(dim);
    }
    return res;
}

std::vector<size_t> ChunkLoader::get_chunk(int dim, char* buffer) {
    std::vector<size_t> res{};
    res.resize(dim,0);
    if(options.use_dictionary()){
        std::string line{};
        size_t disp = 0;
        for(int i = 0 ; i < dim; i++){
            if(!_dictionary.eof())
            {
                std::getline(_dictionary,line);
                if(line != ""){
                res[i] = line.size();
                memcpy(&buffer[disp], line.c_str(), line.size());
                disp += line.size();
                }
                else{
                    i--;
                }
            }
            else{
                break;
            }
        }
    }
    else{
        _generator.generate_chunk(buffer,res.data(),dim);
    }
    
    return res;
}
