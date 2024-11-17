#include "log_engine.hpp"


void LogEngine::trace(LogType type, const std::string& message) {
  //TODO fill the gap
}

LogEngine* LogEngine::instance() {
  static LogEngine instance{};
  return &instance;  
}

LogEngine::LogEngine()
{
  _fd = 0;
}

