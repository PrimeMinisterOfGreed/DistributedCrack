#pragma once
#include <cstring>
#include <options.hpp>


template<typename ... Args>
void _print_impl(const char* type, int verb_limit ,int line, const char* filename,const char * message, Args ... args) {
    if(ARGS.verbosity > verb_limit) {
        char buffer[512]{};
        snprintf(buffer, sizeof(buffer), "[%s][%s:%d]", type,filename, line);
        snprintf(buffer + strlen(buffer), sizeof(buffer) - strlen(buffer), message, args...);
        printf("%s\n", buffer);
    }
}




#define trace(message,...) _print_impl("TRACE",3, __LINE__, __FILE_NAME__, message, ##__VA_ARGS__)
#define info(message,...) _print_impl("INFO".2, __LINE__, __FILE_NAME__, message, ##__VA_ARGS__)
#define debug(message,...) _print_impl("DEBUG",1, __LINE__, __FILE_NAME__, message, ##__VA_ARGS__)
#define error(message,...) _print_impl("ERROR",0, __LINE__, __FILE_NAME__, message, ##__VA_ARGS__)
#define exception(message,...) _print_impl("FATAL",-1, __LINE__, __FILE_NAME__, message, ##__VA_ARGS__)


