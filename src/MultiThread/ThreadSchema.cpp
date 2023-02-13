#include "MultiThread/ThreadSchema.hpp"

ThreadMultiSchema::ThreadMultiSchema(int threads, int initialSequenceLength) : ISchema(), _threads(threads), _mtGenerator(*new MultiThreadStringGenerator(initialSequenceLength))
{
}

void ThreadMultiSchema::Initialize()
{
    for (int i = 0; i < _threads; i++)
        _nodes.push_back(*new ThreadNode(_mtGenerator));
}
