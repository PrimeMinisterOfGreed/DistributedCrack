#include "Node.hpp"

class SchedulerMaster : public MPINode
{
private:
	// Ereditato tramite MPINode
	virtual void Routine() override;
	virtual void Initialize() override;
	virtual void OnBeginRoutine() override;
	virtual void OnEndRoutine() override;
	std::string &_result = *new std::string();
	std::vector<Statistics>& _statistics = *new std::vector<Statistics>();
public:
	SchedulerMaster(boost::mpi::communicator& comm, std::string target) : MPINode(comm, target)
	{
		
	}

	void Report();
};