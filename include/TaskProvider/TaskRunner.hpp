#pragma once
#include "Concepts.hpp"
#include "Nodes/Node.hpp"
#include <list>


template <typename Task> class BaseTaskRunner
{
  protected:
    std::list<ComputeNode<Task>> _requests;
  public:
    virtual void RequestTask(ComputeNode<Task> node){_requests.push_back(node);}
};


template <typename Task> class LocalTaskRunner : public BaseTaskRunner<Task>
{

  protected:
    
  public:
    
};