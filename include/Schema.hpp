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
    virtual void ExecuteSchema() = 0;
    virtual void Initialize(){}
};

class MpiSchema : ISchema
{
  protected:
    boost::mpi::communicator &_comm;

  public:
    MpiSchema(boost::mpi::communicator & comm);
};

class SimpleMasterWorker : public MpiSchema
{
  private:
    int _chunkSize;
    std::string &_target;
  public:
    SimpleMasterWorker(int chunkSize, std::string &target, boost::mpi::communicator& comm);
    void ExecuteSchema() override;
};

class MasterWorkerDistributedGenerator : public MpiSchema
{
  private:
    int _chunkSize;
    std::string &_target;
  public:
    MasterWorkerDistributedGenerator(int chunkSize, std::string &target, boost::mpi::communicator& comm);
    void ExecuteSchema() override;
};

