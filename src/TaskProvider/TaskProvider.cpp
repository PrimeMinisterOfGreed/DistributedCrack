#include "TaskProvider/TaskProvider.hpp"

void BaseTaskProvider::Stop()
{
    for (auto &node : _nodes)
        node.Abort();
}

void BaseTaskProvider::RegisterNode(BaseComputeNode &baseComputeNode)
{
    _nodes.push_back(baseComputeNode);
}
