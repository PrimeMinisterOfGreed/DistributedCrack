#pragma once
#include "Async/async.hpp"
#include "Async/executor.hpp"
#include <queue>
#include <vector>
#include <mpi/mpi.h>

struct node_task {
  std::vector<std::string> _chunk;
  std::string target;
};

class node {
private:
  std::queue<node_task> _tasks{};

public:
  node(MPI::Comm& comm);
  virtual Task Routine();
};

