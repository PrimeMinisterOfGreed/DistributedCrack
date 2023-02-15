#include "MultiThread/ThreadSchema.hpp"

ThreadMultiSchema::ThreadMultiSchema(int threads, int initialSequenceLength,std::string target)
    : ISchema(), _threads(threads), _mtGenerator(*new MultiThreadStringGenerator(initialSequenceLength)), _target(target)
{
}

void ThreadMultiSchema::Initialize()
{
    for (int i = 0; i < _threads; i++)
        _nodes.push_back(*new ThreadNode(_mtGenerator,_target));
}

void ThreadMultiSchema::ExecuteSchema()
{
}
