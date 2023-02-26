#include "MultiThread/ThreadNode.hpp"
#include "DataContainer.hpp"
#include "LogEngine.hpp"
#include "MultiThread/IThreadSchema.hpp"
#include "Nodes/Node.hpp"
#include "Statistics/Event.hpp"
#include "Statistics/EventProcessor.hpp"
#include "Statistics/TimeMachine.hpp"
#include "md5.hpp"
#include <future>
#include <math.h>
#include <string>
#include <thread>

ThreadNode::ThreadNode(IThreadSchema *schema, std::string target, ILogEngine * logger, int process) : NodeHasher(target,logger), _schema(schema),_nodeNum(process)
{
}

void ThreadNode::Execute()
{
    std::thread([this](){Node::Execute();}).detach();
}

void ThreadNode::Initialize()
{
}

void ThreadNode::Routine()
{
    while (!_end)
    {
        auto &chunk = _schema->RequireNextSequence(this);
        std::string *result = new std::string();
        Compute(chunk, result);
        if (result != nullptr)
            _schema->SignalEnd(*result);
    }
}

void ThreadNode::OnBeginRoutine()
{
}

void ThreadNode::OnEndRoutine()
{
    _container->AddResult(*new ExecutionResult{_nodeNum, "MultiThread", GetNodeStats()});

}
