#include "LogEngine.hpp"
class INode
{
public:
	virtual void Execute() = 0;
};

class MPINode : INode
{
private:
protected:
	MPILogEngine* _logger = MPILogEngine::Instance();
public:

	// Ereditato tramite INode
	virtual void Routine();
	virtual void Initialize();
	virtual void Execute() override;
	virtual void OnBeginRoutine();
	virtual void OnEndRoutine();
};