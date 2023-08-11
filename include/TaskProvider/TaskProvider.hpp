#pragma once

#include "Nodes/Node.hpp"
#include "Orchestrators/Orchestrator.hpp"
#include "TaskProvider/Tasks.hpp"
#include <list>


template <typename Task> class ITaskProvider
{
  public:
    virtual Task RequestTask(ComputeNode<Task>& computeNode) = 0;
    virtual void SendTask(Task task) = 0;
};

