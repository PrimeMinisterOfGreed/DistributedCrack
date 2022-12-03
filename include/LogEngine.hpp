#pragma once
#include <boost/mpi.hpp>
#include <fmt/core.h>
#include <fmt/format.h>
#include <iostream>


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

  public:
    // Ereditato tramite ILogEngine
    virtual void Finalize() override;
    // Ereditato tramite ILogEngine
    virtual void Trace(LogType type, std::string &message) override;
    ConsoleLogEngine()
    {
    }
    ConsoleLogEngine(int verbosity) : _verbosity{verbosity}
    {
    }
};

class MPILogEngine : public ILogEngine
{
  private:
    std::istream *_loadStream;
    std::ostream *_saveStream;
    boost::mpi::communicator &_communicator;
    MPILogEngine(boost::mpi::communicator &comm, std::istream *loadStream, std::ostream *saveStream, int verbosity = 1);
    int _verbosity = 1;
    static MPILogEngine *_instance;
    std::ostream &log();

  public:
    static void CreateInstance(boost::mpi::communicator &comm, std::istream *loadStream, std::ostream *saveStream,
                               int verbosity = 1);
    static ILogEngine *Instance();
    virtual void Finalize() override;

    // Ereditato tramite ILogEngine
    virtual void Trace(LogType type, std::string &message) override;
};
