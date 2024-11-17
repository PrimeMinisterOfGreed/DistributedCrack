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

enum class LogType { EXCEPTION, RESULT, INFORMATION, TRANSFER, DEBUG };



struct LogEngine {
private:
  FILE *_fd;

public:

  virtual void trace(LogType type, const std::string& message);
  static LogEngine* instance();
  LogEngine();
};

template<typename T, typename ... Args>
consteval void dbg(const char* fmt, Args...args){
  LogEngine::instance()->trace(LogType::DEBUG, fmt::format(fmt,args...));
}

template<typename T, typename ... Args>
consteval void dbgln(const char* fmt, Args...args){
  LogEngine::instance()->trace(LogType::DEBUG, fmt::format(fmt,args...) + std::endl);
}

template<typename T, typename ... Args>
consteval void info(const char* fmt, Args...args){
  LogEngine::instance()->trace(LogType::INFORMATION, fmt::format(fmt,args...));
}

template<typename T, typename ... Args>
consteval void println(const char* fmt, Args...args){
  LogEngine::instance()->trace(LogType::INFORMATION, fmt::format(fmt,args...) + std::endl);
}

template<typename T, typename ... Args>
consteval void print(const char* fmt, Args...args){
  LogEngine::instance()->trace(LogType::INFORMATION, fmt::format(fmt,args...));
}