#include "ThreadSchema.hpp"

ThreadMultiSchema::ThreadMultiSchema(int threads) : ISchema(), _threads(threads)
{
}

void ThreadMultiSchema::Initialize()
{
    for (int i = 0; i < _threads; i++)
        _nodes.push_back(*new ThreadNode());
}
