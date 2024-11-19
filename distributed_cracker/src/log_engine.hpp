#pragma once
#include <fmt/core.h>
#include <fmt/format.h>
#include <iostream>
#ifndef MAX_LOG_SIZE
#define MAX_LOG_SIZE 1024
#endif
#ifdef DEBUG
constexpr bool _is_debug = true;
#else 
constexpr bool _is_debug = false;
#endif


constexpr size_t max_log_size = MAX_LOG_SIZE;


#define dbg(message,...) LogEngine::instance()->trace(__FILE_NAME__,__LINE__,fmt::format(message __VA_OPT__(,__VA_ARGS__)),false)
#define dbgln(message,...) LogEngine::instance()->trace(__FILE_NAME__,__LINE__,fmt::format(message __VA_OPT__(,__VA_ARGS__)),true)
#define print(message,...) LogEngine::instance()->trace(__FILE_NAME__,__LINE__,fmt::format(message __VA_OPT__(,__VA_ARGS__)),false)
#define println(message,...) LogEngine::instance()->trace(__FILE_NAME__,__LINE__,fmt::format(message __VA_OPT__(,__VA_ARGS__)),false)
struct LogEngine {
private:
  FILE *_fd;

public:

  virtual void trace(const char* filename, int line, const std::string& message, bool newline);
  static LogEngine* instance();
  LogEngine();
};

