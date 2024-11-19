#include "log_engine.hpp"




void LogEngine::trace(const char* filename, int line, const std::string& message, bool newline) {
  printf("[%s:%d]%s %s",filename,line,message.c_str(),newline?"\r\n":"");
}

LogEngine* LogEngine::instance() {
  static LogEngine instance{};
  return &instance;  
}

LogEngine::LogEngine()
{
  _fd = 0;
}

