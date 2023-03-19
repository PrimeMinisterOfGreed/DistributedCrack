#pragma once

#include <boost/mpi/communicator.hpp>
using communicator = boost::mpi::communicator;



template<typename Task>
class MPITaskReceiver
{
  protected:
    communicator &_comm;

  public:
    Task &operator()()
    {
        
    }

    
};


class MPITaskGenerator
{
  protected:
    communicator &_comm;

  public:
    MPITaskGenerator();
};

