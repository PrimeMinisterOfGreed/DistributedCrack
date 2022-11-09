#pragma once
#include "Node.hpp"
#include "StringGenerator.hpp"
class AddressableWorker : public MPINode
{
private:
	// Ereditato tramite MPINode
	virtual void Routine() override;

	virtual void Initialize() override;

	virtual void OnBeginRoutine() override;

	virtual void OnEndRoutine() override;

	AssignedSequenceGenerator* _generator;
	
public:
	AddressableWorker(boost::mpi::communicator& comm): MPINode(comm){}
};