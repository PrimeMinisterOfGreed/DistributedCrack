#include "MultiThread/ThreadNode.hpp"
#include "Nodes/Node.hpp"
#include "Statistics/Event.hpp"
#include "Statistics/EventProcessor.hpp"
#include "Statistics/TimeMachine.hpp"
#include "md5.hpp"


ThreadNode::ThreadNode( ThreadMultiSchema & _schema, std::string target)
    : NodeHasher(target), _schema(_schema)
{
}



void ThreadNode::Initialize()
{
    
}

void ThreadNode::Routine()
{
    
}

void ThreadNode::OnBeginRoutine()
{
}

void ThreadNode::OnEndRoutine()
{
}
