#include "MultiThread/ThreadSchema.hpp"
#include "LogEngine.hpp"
#include "Nodes/Node.hpp"
#include <future>
#include <vector>

ThreadMultiSchema::ThreadMultiSchema(int threads, int initialSequenceLength, std::string target, int initialChunkSize, ILogEngine * logger)
    : _threads(threads), _mtGenerator(*new MultiThreadStringGenerator(initialSequenceLength)), _target(target), _logger(logger), _initialChunkSize(initialChunkSize)
{
}

void ThreadMultiSchema::Initialize()
{
    for (int i = 0; i < _threads; i++)
        _nodes.push_back(*new ThreadNode(this, _target,_logger));
}

void ThreadMultiSchema::ExecuteSchema()
{
    for (auto &node : _nodes)
    {
        node.Execute();
    }
    _waitEnd.WaitOne();
}

std::vector<std::string> &ThreadMultiSchema::RequireNextSequence(ThreadNode *requiringNode)
{
    static int chunkSize = _initialChunkSize;
    if (requiringNode != nullptr)
    {
        if (_lastRunStat.count(requiringNode))
        {
            auto &prevStat = _lastRunStat.at(requiringNode);
            auto &actualStat = requiringNode->GetNodeStats();
            double strainCoeff = actualStat.throughput - prevStat.throughput;
            chunkSize += strainCoeff>=0? 1:-1;
        }
            _lastRunStat[requiringNode] = Statistics(requiringNode->GetNodeStats());
    }
    _logger->TraceInformation("Current Index{}", _mtGenerator.GetCurrentIndex());
    return _mtGenerator.SafeGenerateChunk(chunkSize);
}

void ThreadMultiSchema::SignalEnd(std::string &result)
{
    _result = result;
    for (auto &node : _nodes)
        node.ForceEnd();
    _waitEnd.Set();
}
