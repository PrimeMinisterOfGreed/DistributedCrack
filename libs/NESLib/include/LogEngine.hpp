#pragma once

#include <fmt/core.h>
#include <fmt/format.h>
#include <iostream>
#include <sstream>
#include <string>

enum class LogType
{
    EXCEPTION,
    RESULT,
    INFORMATION,
    TRANSFER,
    DEBUG
};

template <typename... Args> std::string &makeformat(const char *format, Args... args)
{
    return *new std::string(fmt::vformat(std::string(format), fmt::make_format_args(args...)));
}

class ILogEngine
{
  public:
    virtual void Finalize() = 0;
    virtual void Trace(LogType type, std::string &message) = 0;

    template <typename... Args> void TraceException(const char *format, Args... args)
    {
        Trace(LogType::EXCEPTION, makeformat(format, args...));
    }

    template <typename... Args> void TraceInformation(const char *format, Args... args)
    {
        Trace(LogType::INFORMATION, makeformat(format, args...));
    }

    template <typename... Args> void TraceTransfer(const char *format, Args... args)
    {
        Trace(LogType::TRANSFER, makeformat(format, args...));
    }

    template <typename... Args> void TraceResult(const char *format, Args... args)
    {
        Trace(LogType::RESULT, makeformat(format, args...));
    }

    template <typename... Args> void TraceDebug(const char *format, Args... args)
    {
        Trace(LogType::DEBUG, makeformat(format, args...));
    }
};

class ConsoleLogEngine : public ILogEngine
{
  private:
    int _verbosity = 1;
    std::stringstream _buffer;
    std::string _logFile;
    static ConsoleLogEngine *_instance;
    ConsoleLogEngine()
    {
    }
    ConsoleLogEngine(int verbosity, std::string logFile) : _verbosity{verbosity}, _logFile{logFile}
    {
    }

  public:
    virtual void Finalize() override;
    virtual void Trace(LogType type, std::string &message) override;

    static void CreateInstance(int verbosity, std::string logFile)
    {
        _instance = new ConsoleLogEngine(verbosity, logFile);
    }
    static ConsoleLogEngine *Instance()
    {
        return _instance;
    }
};

template <char Divisor>
constexpr const char *PrintDivisor()
{
    char * buffer = new char[24];
    for (int i = 0; i < 24; i++)
        buffer[i] = Divisor;
    return buffer;
}


