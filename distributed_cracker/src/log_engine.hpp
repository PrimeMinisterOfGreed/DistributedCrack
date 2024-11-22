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

#if DEBUG_MODE
#define dbg(message,...) LogEngine::instance()->trace(__FILE_NAME__,__LINE__,fmt::format(message __VA_OPT__(,__VA_ARGS__)))
#define dbgln(message,...) dbg("{}\n",fmt::format(message __VA_OPT__(,__VA_ARGS__)))
#else 
#define dbg(message,...)
#define dbgln(message,...)
#endif

#define print(message,...) LogEngine::instance()->trace(__FILE_NAME__,__LINE__,fmt::format(message __VA_OPT__(,__VA_ARGS__)))
#define println(message,...) print("{}\n",fmt::format(message __VA_OPT__(,__VA_ARGS__)))
struct LogEngine {
private:
  FILE *_fd;

public:

  virtual void trace(const char* filename, int line, const std::string& message);
  static LogEngine* instance();
  LogEngine();
};

