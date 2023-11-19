#pragma once
#include "Async/Async.hpp"
#include "Async/Executor.hpp"
#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <queue>
#include <vector>

struct NodeTask {
  std::vector<std::string> _chunk;
  std::string target;
};

class Node {
private:
  boost::mpi::communicator _comm;
  std::queue<NodeTask> _tasks{};

public:
  Node(boost::mpi::communicator comm);
  // FuturePtr<NodeTask> WaitTask();
  // FuturePtr<void> SignalReady();
  // FuturePtr<void> ComputeTask();
  virtual Task Routine();
};

class GeneratorNode : public Node {

public:
  GeneratorNode(boost::mpi::communicator comm);
  // FuturePtr<void> SendTask(NodeTask task, int node);
  virtual Task Routine() override;
};