#include "log_engine.hpp"
#include <variant>

void LogEngine::trace(const char *filename, int line,
                      const std::string &message) {
  static bool inlined = false;
  if(message.find('\n') == std::variant_npos){
    if(!inlined){
      printf("[%s:%d]%s",filename,line,message.c_str());
      inlined = true;
    }
    else{
      printf("%s",message.c_str());
    }
  } else {
    if(inlined){
      printf("\n");
    }
    printf("[%s:%d]%s", filename, line, message.c_str());
    inlined = false;
  }
}

LogEngine *LogEngine::instance() {
  static LogEngine instance{};
  return &instance;
}

LogEngine::LogEngine() { _fd = 0; }
