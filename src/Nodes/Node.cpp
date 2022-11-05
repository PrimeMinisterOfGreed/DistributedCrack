#include "Nodes/Node.hpp"

void Node::Execute()
{
	try
	{
		Initialize();
		BeginRoutine();
		Routine();
		EndRoutine();
	}
	catch (std::exception ex)
	{
		_logger->TraceException() << ex.what() << std::endl;
		_logger->Finalize();
	}
}



void Node::BeginRoutine()
{
	_logger->TraceInformation() << "Routine Setup" << std::endl;
	try
	{
		OnBeginRoutine();
	}
	catch (std::exception e)
	{
		_logger->TraceException() << "Exception during routine setup:" << e.what() << std::endl;
		throw;
	}
	_logger->TraceInformation() << "Routine Setup completed" << std::endl;
}

void Node::EndRoutine()
{
	_logger->TraceInformation() << "Ending Routine" << std::endl;
	try
	{
		OnEndRoutine();
	}
	catch (std::exception e)
	{
		_logger->TraceException() << "Exception during routine ending: " << e.what() << std::endl;
		throw;
	}
	_logger->TraceInformation() << "Routine end done" << std::endl;
}

void Node::ExecuteRoutine()
{
	_logger->TraceInformation() << "Routine Execution" << std::endl;
	try
	{
		Routine();
	}
	catch (std::exception e)
	{
		_logger->TraceException() << "Exception during routine execution: " << e.what() << std::endl;
		throw;
	}
	_logger->TraceInformation() << "Routine execution done" << std::endl;
}


