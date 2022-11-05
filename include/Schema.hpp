#pragma once
#include <boost/concept_check.hpp>
#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/request.hpp>
#include <functional>
#include <iterator>
#include <string>
#include <vector>
#include "Functions.hpp"
enum
{
    MESSAGE,
    WORK,
    TERMINATE,
    FOUND
};


class ISchema
{
  public:
    virtual void ExecuteSchema(boost::mpi::communicator &comm) = 0;
};

class SimpleMasterWorker : ISchema
{
  private:
    void Master(boost::mpi::communicator &comm);
    void Worker(boost::mpi::communicator &comm);
    int _chunkSize;
    std::string &_target;

  public:
    SimpleMasterWorker(int chunkSize, std::string &target);
    void ExecuteSchema(boost::mpi::communicator &comm) override;
};

class MasterWorkerDistributedGenerator : ISchema
{
  private:
    void Master(boost::mpi::communicator &comm);
    void Worker(boost::mpi::communicator &comm);
    int _chunkSize;
    std::string &_target;

  public:
    MasterWorkerDistributedGenerator(int chunkSize, std::string &target);
    void ExecuteSchema(boost::mpi::communicator &comm) override;
};
