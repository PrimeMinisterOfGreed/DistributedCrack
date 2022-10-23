#pragma once
#include "Schema.hpp"
#include <boost/mpi/communicator.hpp>

class RingPipeLine : ISchema
{
  private:
    void MasterNode(boost::mpi::communicator &comm);
    void Node(boost::mpi::communicator &comm);
    int _chunkSize;
    std::string &_target;

  public:
    RingPipeLine(int chunkSize, std::string &target);
    void ExecuteSchema(boost::mpi::communicator &comm);
};
