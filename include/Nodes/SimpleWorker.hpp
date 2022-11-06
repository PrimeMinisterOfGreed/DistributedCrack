#include "Node.hpp"

class SimpleWorker : public MPINode
{
private:
	std::string _target;
public:
	// Ereditato tramite Node
	virtual void Routine() override;
	virtual void Initialize() override;
	virtual void OnBeginRoutine() override;
	virtual void OnEndRoutine() override;
	SimpleWorker(boost::mpi::communicator comm);
};