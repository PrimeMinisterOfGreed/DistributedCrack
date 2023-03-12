#pragma once

#include "Nodes/Node.hpp"
#include "Orchestrators/Orchestrator.hpp"
#include "TaskProvider/Tasks.hpp"
#include <list>

class ITaskProvider
{
  public:
    virtual void RegisterNode(BaseComputeNode &baseComputeNode) = 0;
    virtual void Stop() = 0;
};



class BaseTaskProvider : public ITaskProvider
{
  protected:
    std::list<BaseComputeNode> _nodes{};
  public:
    void Stop() override;
    void RegisterNode(BaseComputeNode &baseComputeNode) override;
};



