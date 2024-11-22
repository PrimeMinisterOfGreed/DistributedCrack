#include "chunk_loader.hpp"
#include "options_bag.hpp"
#include <cstring>
#include <filesystem>
auto options = ProgramOptions::instance();

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
    if (options->dictionary != _state.filename) {
      _state.filename << options->dictionary;
      _state.actualseq = 0;
    }
    if(options->use_dictionary()){
        _dictionary.open(_state.filename);
    }
    else{
        _generator.assign_address(_state.actualseq);
    }
    _state_store.write(reinterpret_cast<char*>(&_state), sizeof(LoaderState));
}

std::vector<std::string> ChunkLoader::get_chunk(int dim) {
    std::vector<std::string> res{};
    if(options->use_dictionary()){
        std::string line{};
        for(int i = 0; i < dim; i++){
            std::getline(_dictionary,line);
            if(line != "")
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
