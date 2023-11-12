#pragma once
#include "Async/Async.hpp"
#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <vector>

struct NodeTask {
  std::vector<std::string> _chunk;
  std::string target;
};

class Node {
private:
  boost::mpi::communicator _comm;

public:
  sptr<Future<NodeTask>> WaitTask();
};