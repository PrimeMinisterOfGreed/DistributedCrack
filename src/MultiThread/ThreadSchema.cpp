#include "MultiThread/ThreadSchema.hpp"
#include <vector>

ThreadMultiSchema::ThreadMultiSchema(int threads, int initialSequenceLength, std::string target, int initialChunkSize)
    : ISchema(), _threads(threads), _mtGenerator(*new MultiThreadStringGenerator(initialSequenceLength)),
      _target(target)
{
}

void ThreadMultiSchema::Initialize()
{
    for (int i = 0; i < _threads; i++)
        _nodes.push_back(*new ThreadNode(*this, _target));
}

void ThreadMultiSchema::ExecuteSchema()
{
}

std::vector<std::string>& ThreadMultiSchema::RequireNextSequence(const ThreadNode *requiringNode)
{
    int chunkSize = _initialChunkSize;
    if (requiringNode != nullptr)
    {
        if (_lastRunStat.count(requiringNode))
        {
            auto &prevStat = _lastRunStat.at(requiringNode);
            auto &actualStat = requiringNode->GetNodeStats();
            double strainCoeff = 0;
        }
        else
        {
            
        }
    }
    return _mtGenerator.SafeGenerateChunk(chunkSize);
}


