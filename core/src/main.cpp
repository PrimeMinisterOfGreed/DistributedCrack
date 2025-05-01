#include "options.hpp"


int main(int argc, char ** argv){
    //get launch.toml from argv[1]
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <config_file>\n", argv[0]);
        return 1;
    }
    const char* config_file = argv[1]; 
    ARGS =  options::load_from_file(config_file).unwrap();
}