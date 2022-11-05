#include "Nodes/Node.hpp"
#include <boost/mpi.hpp>

class SimpleMaster : public MPINode
{
  private:
    boost::mpi::communicator _comm;
    std::string _target;
  public:
    SimpleMaster(boost::mpi::communicator comm, std::string target);
    virtual void OnEndRoutine() override;
    virtual void Initialize() override;
    virtual void Routine() override;
    virtual void OnBeginRoutine() override;
};