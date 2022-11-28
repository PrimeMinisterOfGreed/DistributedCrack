#pragma once 
#include <boost/mpi.hpp>
#include <iostream>
#include <format>
class ILogEngine
{
public:
	virtual void Finalize() = 0;
	virtual void TraceException(std::string& message) = 0;
	virtual void TraceInformation(std::string& message) = 0;
	virtual void TraceTransfer(std::string& message) = 0;
	virtual void TraceResult(std::string& message) = 0;

	template<typename ...Args>
	void TraceException(const char* format, Args... args)
	{
		TraceException(*new std::string(std::vformat(std::string_view(format), std::make_format_args(args...))));
	}

	template<typename ...Args>
	void TraceInformation(const char* format, Args...args)
	{
		TraceException(*new std::string(std::vformat(std::string_view(format), std::make_format_args(args...))));
	}

	template <typename ...Args>
	void TraceTransfer(const char* format, Args...args)
	{
		TraceException(*new std::string(std::vformat(std::string_view(format), std::make_format_args(args...))));
	}

	template<typename ...Args>
	void TraceResult(const char* format, Args...args)
	{
		TraceException(*new std::string(std::vformat(std::string_view(format), std::make_format_args(args...))));
	}
};

class ConsoleLogEngine : public ILogEngine
{

public:
	// Ereditato tramite ILogEngine
	virtual void Finalize() override;

	// Ereditato tramite ILogEngine
	virtual void TraceException(std::string& message) override;
	virtual void TraceInformation(std::string& message) override;
	virtual void TraceTransfer(std::string& message) override;
	virtual void TraceResult(std::string& message) override;
};

class MPILogEngine : public ILogEngine
{
private:
	std::istream* _loadStream;
	std::ostream* _saveStream;
	boost::mpi::communicator& _communicator;
	MPILogEngine(boost::mpi::communicator& comm, std::istream* loadStream, std::ostream* saveStream, int verbosity);
	int _verbosity;
	static MPILogEngine* _instance;
	std::ostream& log();
public:
	static void CreateInstance(boost::mpi::communicator& comm, std::istream* loadStream, std::ostream* saveStream, int verbosity = 0);
	static ILogEngine* Instance();
	virtual void Finalize() override;


	// Ereditato tramite ILogEngine
	virtual void TraceException(std::string& message) override;

	virtual void TraceInformation(std::string& message) override;

	virtual void TraceTransfer(std::string& message) override;

	virtual void TraceResult(std::string& message) override;

};


