#include "MultiThread/ThreadNode.hpp"
#include "Statistics/Event.hpp"
#include "Statistics/EventProcessor.hpp"
#include "Statistics/TimeMachine.hpp"
#include "md5.hpp"
ThreadNode::ThreadNode(ISequenceGenerator &generator, std::string target)
    : Node(), _generator(generator), _target(target), _stopWatch(*new StopWatch())
{
}

bool ThreadNode::Compute(std::vector<std::string> &chunk, std::string target, std::string *result)
{
    
    auto &res = _stopWatch.RecordEvent([result, target, chunk](Event &evt) {
        for (auto &value : chunk)
        {
            if (md5(value) == target)
            {
                *result = value;
                return true;
            }
            evt.completitions++;
        }
    });
    _processor.AddEvent(res);
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
