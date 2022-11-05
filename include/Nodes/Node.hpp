#include "LogEngine.hpp"
class INode
{
public:
	virtual void Execute() = 0;
};

class Node: INode
{
private:
	void BeginRoutine();
	void EndRoutine();
	void ExecuteRoutine();
protected:
	MPILogEngine* _logger = MPILogEngine::Instance();
public:
	virtual void Routine() = 0;
	virtual void Initialize() = 0;
	virtual void OnBeginRoutine() = 0;
	virtual void OnEndRoutine() = 0;
	virtual void Execute() override;
};

