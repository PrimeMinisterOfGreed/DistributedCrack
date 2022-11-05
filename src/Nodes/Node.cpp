#include "Nodes/Node.hpp"
void MPINode::Routine()
{

}

void MPINode::Initialize()
{
	_logger->TraceInformation() << "Initializing node" << std::endl;
}

void MPINode::Execute()
{
	try
	{
		Initialize();
		OnBeginRoutine();
		Routine();
		OnEndRoutine();
	}
	catch (std::exception ex)
	{
		_logger->TraceException() << ex.what() << std::endl;
		_logger->Finalize();
	}
}

void MPINode::OnBeginRoutine()
{
	_logger->TraceInformation() << "Routine Setup" << std::endl;
}

void MPINode::OnEndRoutine()
{
	_logger->TraceInformation() << "End" << std::endl;
	_logger->Finalize();
}
